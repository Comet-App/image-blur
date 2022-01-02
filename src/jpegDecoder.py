from PIL import Image, ImageFilter
from scipy import fftpack
import numpy
from bitstream import BitStream
from numpy import *
import huffmanEncode
import sys
from jpegHelper import config

zigzagOrder = config.zigzagOrder
std_luminance_quant_tbl = config.std_luminance_quant_tbl.reshape([8,8])
std_chrominance_quant_tbl = config.std_chrominance_quant_tbl.reshape([8,8])
preview_width = config.preview_width
preview_height = config.preview_height
quality = config.quality_img
OFFSET = config.offset
DEBUG_MODE = config.debug_mode

def main():
    global quality, DEBUG_MODE

    if(len(sys.argv)<3 or len(sys.argv)>4):
        print('inputDatFileName outputJPEGFilename DEBUG_MODE[0 | 1]')
        print('example:')
        print('HEX_STRING ./output.jpg')
        return

    srcFile = sys.argv[1]
    outputJPEGFile = sys.argv[2]

    if(len(sys.argv)==4):
        if sys.argv[3] == '0' or sys.argv[3] == '1':
            DEBUG_MODE = int(sys.argv[3])

    # Read encoded data
    data = bytes.fromhex(srcFile)

    numpy.set_printoptions(threshold=numpy.inf)
    tw = format(data[-2], '08b')
    th = format(data[-1], '08b')

    orgImWidth = round((int(tw[:5],2) + int(tw[5:8],2)*0.1)*OFFSET)
    orgImHeight = round((int(th[:5],2) + int(th[5:8],2)*0.1)*OFFSET)
    srcImageWidth = preview_width  
    srcImageHeight = preview_height
    
    print('srcImageWidth = %d srcImageHeight = %d' % (srcImageWidth, srcImageHeight))

    imageWidth = srcImageWidth
    imageHeight = srcImageHeight
    # add width and height to %8==0
    if (srcImageWidth % 8 != 0):
        imageWidth = srcImageWidth // 8 * 8 + 8
    if (srcImageHeight % 8 != 0):
        imageHeight = srcImageHeight // 8 * 8 + 8

    print('added to: ', imageWidth, imageHeight)


    if(quality <= 0):
        quality = 1
    if(quality > 100):
        quality = 100
    if(quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
    luminanceQuantTbl = numpy.array(numpy.floor((std_luminance_quant_tbl * qualityScale + 50) / 100))
    luminanceQuantTbl[luminanceQuantTbl == 0] = 1
    luminanceQuantTbl[luminanceQuantTbl > 255] = 255
    luminanceQuantTbl = luminanceQuantTbl.reshape([8, 8]).astype(int)
    if DEBUG_MODE == 1:
        print('luminanceQuantTbl:\n', luminanceQuantTbl)
    chrominanceQuantTbl = numpy.array(numpy.floor((std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl[chrominanceQuantTbl > 255] = 255
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)
    if DEBUG_MODE == 1:
        print('chrominanceQuantTbl:\n', chrominanceQuantTbl)
    blockSum = imageWidth // 8 * imageHeight // 8

    if DEBUG_MODE == 1:
        print('blockSum = ', blockSum)

    sosBitStream = BitStream()

    jpegFile = open(outputJPEGFile, 'wb+')
    # write jpeg header
    jpegFile.write(huffmanEncode.hexToBytes('FFD8FFE000104A46494600010100000100010000'))
    # write y Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004300'))
    luminanceQuantTbl = luminanceQuantTbl.reshape([64])
    jpegFile.write(bytes(luminanceQuantTbl.tolist()))
    # write u/v Quantization Table
    jpegFile.write(huffmanEncode.hexToBytes('FFDB004301'))
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([64])
    jpegFile.write(bytes(chrominanceQuantTbl.tolist()))
    # write height and width
    jpegFile.write(huffmanEncode.hexToBytes('FFC0001108'))
    hHex = hex(srcImageHeight)[2:]
    while len(hHex) != 4:
        hHex = '0' + hHex

    jpegFile.write(huffmanEncode.hexToBytes(hHex))

    wHex = hex(srcImageWidth)[2:]
    while len(wHex) != 4:
        wHex = '0' + wHex

    jpegFile.write(huffmanEncode.hexToBytes(wHex))

    # 03    01 11 00    02 11 01    03 11 01
    # 1：1	01 11 00	02 11 01	03 11 01
    # 1：2	01 21 00	02 11 01	03 11 01
    # 1：4	01 22 00	02 11 01	03 11 01
    # write Subsamp
    jpegFile.write(huffmanEncode.hexToBytes('03011100021101031101'))

    #write huffman table
    jpegFile.write(huffmanEncode.hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))
    # SOS Start of Scan
    # yDC yAC uDC uAC vDC vAC
    sosLength = sosBitStream.__len__()
    filledNum = 8 - sosLength % 8
    if(filledNum!=0):
        sosBitStream.write(numpy.ones([filledNum]).tolist(),bool)

    jpegFile.write(bytes([255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0])) # FF DA 00 0C 03 01 00 02 11 03 11 00 3F 00

    # write encoded data --- last two bytes are for width and height
    sosBytes = data[:-2] 
    for i in range(len(sosBytes)):
        jpegFile.write(bytes([sosBytes[i]])) # FF to FF 00
        if(sosBytes[i]==255):
            jpegFile.write(bytes([0]))

    # write end symbol
    jpegFile.write(bytes([255,217])) # FF D9
    jpegFile.close()

    im = Image.open(outputJPEGFile)
    im = im.resize((orgImWidth,orgImHeight))
    im_save = im.filter(ImageFilter.GaussianBlur(4))
    im_save.save(outputJPEGFile)

if __name__ == '__main__':
    main()



