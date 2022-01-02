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


def roundOfSize(width, height):
    w = width//OFFSET
    width = round(width/OFFSET,1)
    h = height//OFFSET
    height = round(height/OFFSET,1)
    
    wp = (width - w)*10
    hp = (height - h)*10
    return int(w), int(wp), int(h), int(hp)

def decimalToBin(num, arr, c):
    if c < 0 or num == 0:
        return arr
    if num>=1:
        arr[c] = num % 2
        num = num // 2
        c = c - 1
        return decimalToBin(num, arr, c)

def main():
    global quality

    if(len(sys.argv)!=2):
        print('inputJpgFileName')
        print('example:')
        print('./lena.jpg')
        return

    srcFileName = sys.argv[1]
    
    numpy.set_printoptions(threshold=numpy.inf)
    srcImage = Image.open(srcFileName)
    orgImWidth, orgImHeight = srcImage.size
    print('orgImageWidth = %d orgImageHeight = %d' % (orgImWidth, orgImHeight))
    srcImage = srcImage.resize((preview_width,preview_height))
    srcImageWidth, srcImageHeight = srcImage.size
    print('srcImageWidth = %d srcImageHeight = %d' % (srcImageWidth, srcImageHeight))
    print('srcImage info:\n', srcImage)
    srcImageMatrix = numpy.asarray(srcImage)

    imageWidth = srcImageWidth
    imageHeight = srcImageHeight
    # add width and height to %8==0
    if (srcImageWidth % 8 != 0):
        imageWidth = srcImageWidth // 8 * 8 + 8
    if (srcImageHeight % 8 != 0):
        imageHeight = srcImageHeight // 8 * 8 + 8

    if DEBUG_MODE == 1:
        print('added to: ', imageWidth, imageHeight)

    # copy data from srcImageMatrix to addedImageMatrix
    addedImageMatrix = numpy.zeros((imageHeight, imageWidth, 3), dtype=numpy.uint8)
    for y in range(srcImageHeight):
        for x in range(srcImageWidth):
            addedImageMatrix[y][x] = srcImageMatrix[y][x]

    # split y u v
    yImage,uImage,vImage = Image.fromarray(addedImageMatrix).convert('YCbCr').split()

    yImageMatrix = numpy.asarray(yImage).astype(int)
    uImageMatrix = numpy.asarray(uImage).astype(int)
    vImageMatrix = numpy.asarray(vImage).astype(int)
    if(DEBUG_MODE==1):
        print('yImageMatrix:\n', yImageMatrix)
        print('uImageMatrix:\n', uImageMatrix)
        print('vImageMatrix:\n', vImageMatrix)

    yImageMatrix = yImageMatrix - 127
    uImageMatrix = uImageMatrix - 127
    vImageMatrix = vImageMatrix - 127


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
    #print('luminanceQuantTbl:\n', luminanceQuantTbl)
    chrominanceQuantTbl = numpy.array(numpy.floor((std_chrominance_quant_tbl * qualityScale + 50) / 100))
    chrominanceQuantTbl[chrominanceQuantTbl == 0] = 1
    chrominanceQuantTbl[chrominanceQuantTbl > 255] = 255
    chrominanceQuantTbl = chrominanceQuantTbl.reshape([8, 8]).astype(int)
    #print('chrominanceQuantTbl:\n', chrominanceQuantTbl)
    blockSum = imageWidth // 8 * imageHeight // 8

    yDC = numpy.zeros([blockSum], dtype=int)
    uDC = numpy.zeros([blockSum], dtype=int)
    vDC = numpy.zeros([blockSum], dtype=int)
    dyDC = numpy.zeros([blockSum], dtype=int)
    duDC = numpy.zeros([blockSum], dtype=int)
    dvDC = numpy.zeros([blockSum], dtype=int)

    if DEBUG_MODE == 1:
        print('blockSum = ', blockSum)

    sosBitStream = BitStream()

    blockNum = 0
    for y in range(0, imageHeight, 8):
        for x in range(0, imageWidth, 8):
            #print('block (y,x): ',y, x, ' -> ', y + 8, x + 8)
            yDctMatrix = fftpack.dct(fftpack.dct(yImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            uDctMatrix = fftpack.dct(fftpack.dct(uImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            vDctMatrix = fftpack.dct(fftpack.dct(vImageMatrix[y:y + 8, x:x + 8], norm='ortho').T, norm='ortho').T
            
            yQuantMatrix = numpy.rint(yDctMatrix / luminanceQuantTbl)
            uQuantMatrix = numpy.rint(uDctMatrix / chrominanceQuantTbl)
            vQuantMatrix = numpy.rint(vDctMatrix / chrominanceQuantTbl)

            yZCode = yQuantMatrix.reshape([64])[zigzagOrder]
            uZCode = uQuantMatrix.reshape([64])[zigzagOrder]
            vZCode = vQuantMatrix.reshape([64])[zigzagOrder]
            yZCode = yZCode.astype(int)
            uZCode = uZCode.astype(int)
            vZCode = vZCode.astype(int)

            yDC[blockNum] = yZCode[0]
            uDC[blockNum] = uZCode[0]
            vDC[blockNum] = vZCode[0]

            if(blockNum==0):
                dyDC[blockNum] = yDC[blockNum]
                duDC[blockNum] = uDC[blockNum]
                dvDC[blockNum] = vDC[blockNum]
            else:
                dyDC[blockNum] = yDC[blockNum] - yDC[blockNum-1]
                duDC[blockNum] = uDC[blockNum] - uDC[blockNum-1]
                dvDC[blockNum] = vDC[blockNum] - vDC[blockNum-1]


            # huffman encode https://www.impulseadventure.com/photo/jpeg-huffman-coding.html
            # encode yDC
            if(DEBUG_MODE==1):
                print("encode dyDC:",dyDC[blockNum])
            sosBitStream.write(huffmanEncode.encodeDCToBoolList(dyDC[blockNum],1, DEBUG_MODE),bool)
            # encode yAC
            if (DEBUG_MODE == 1):
                print("encode yAC:", yZCode[1:])
            huffmanEncode.encodeACBlock(sosBitStream, yZCode[1:], 1, DEBUG_MODE)

            # encode uDC
            if(DEBUG_MODE==1):
                print("encode duDC:",duDC[blockNum])
            sosBitStream.write(huffmanEncode.encodeDCToBoolList(duDC[blockNum],0, DEBUG_MODE),bool)
            # encode uAC
            if (DEBUG_MODE == 1):
                print("encode uAC:", uZCode[1:])
            huffmanEncode.encodeACBlock(sosBitStream, uZCode[1:], 0, DEBUG_MODE)

            # encode vDC
            if(DEBUG_MODE==1):
                print("encode dvDC:",dvDC[blockNum])
            sosBitStream.write(huffmanEncode.encodeDCToBoolList(dvDC[blockNum],0, DEBUG_MODE),bool)
            # encode uAC
            if (DEBUG_MODE == 1):
                print("encode vAC:", vZCode[1:])
            huffmanEncode.encodeACBlock(sosBitStream, vZCode[1:], 0, DEBUG_MODE)

            blockNum = blockNum + 1

    sosLength = sosBitStream.__len__()
    filledNum = 8 - sosLength % 8
    if(filledNum!=0):
        sosBitStream.write(numpy.ones([filledNum]).tolist(),bool)
            
    w, wp, h, hp = roundOfSize(orgImWidth, orgImHeight)
    error = False
    if(w>31):
        error = True
        print("exceed width:", orgImWidth)
    if(h>31):
        error = True
        print("exceed height:", orgImHeight)
    
    if(h==31 and hp>7):
        hp = 7
    else:
        if(h<31 and hp>7):
            h = h + 1
            hp = 0
    if(w==31 and wp>7):
        wp = 7
    else:
        if(w<31 and wp>7):
            w = w + 1
            wp = 0

    if(not error):
        fivebittemp, threebittemp = [0,0,0,0,0], [0,0,0]
        binw = decimalToBin(w, fivebittemp, len(fivebittemp)-1)
        binwp = decimalToBin(wp, threebittemp, len(threebittemp)-1)

        fivebittemp, threebittemp = [0,0,0,0,0], [0,0,0]
        binh = decimalToBin(h, fivebittemp, len(fivebittemp)-1)
        binhp = decimalToBin(hp, threebittemp, len(threebittemp)-1)
        
        sosBitStream.write(binw+binwp+binh+binhp,bool)
        sosBytes = sosBitStream.read(bytes)
        #print(sosBytes[-2], sosBytes[-1])
        if DEBUG_MODE == 1:
            print(w,wp,h,hp)
        print("Byte Length {0}".format(len(sosBytes)))
        print(sosBytes.hex())
        

if __name__ == '__main__':
    main()
    



