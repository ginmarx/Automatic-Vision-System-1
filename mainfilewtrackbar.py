from pypylon import pylon
import time
import numpy as np
import cv2
import os
#import concurrent.futures
import threading
import datetime
import pathlib
import pandas as pd
#import WaterLevelProcess as WLP
import firstsetup as fst
#import BarcodeProcess as BCP

def FirstProcess(st, ser1, path, camera):
    BottleCount = int(ser1.readline())
    time.sleep(.96)
    imMode1 = fst.capture(camera, st['1ple'], st['1plgm'], st['1plgn'], st['1plds'])
    imMode2 = fst.capture(camera, st['2ple'], st['2plgm'], st['2plgn'], st['2plds'])
    imMode3 = fst.capture(camera, st['3ple'], st['3plgm'], st['3plgn'], st['3plds'])
    try:
        # start=time.perf_counter()
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     p2 = executor.submit(WLP.WaterLevelProcess, imMode2.Array, BottleCount, path)
        #     p1 = executor.submit(BCP.Barcode, imMode1.Array)
        stop = time.perf_counter()
        # print(round(stop - start, 2))
        # con2, p2img, p2data = p2.result()
        # con1, p1img, p1data = p1.result()

        pathlib.Path(path + '\\Mode1\\').mkdir(parents=True,exist_ok=True)
        pathlib.Path(path + '\\Mode2\\').mkdir(parents=True,exist_ok=True)
        pathlib.Path(path + '\\Mode3\\').mkdir(parents=True,exist_ok=True)
        pathlib.Path(path + '\\barcode\\').mkdir(parents=True,exist_ok=True)

        name1 = os.path.join(path,'Mode1', str(BottleCount) + '.tiff')
        name2 = os.path.join(path,'Mode2', str(BottleCount) + '.tiff')
        name3 = os.path.join(path,'Mode3', str(BottleCount) + '.tiff')
        name4 = os.path.join(path,'barcode', str(BottleCount) + '.tiff')


        try:
            print('Saving images...')
            cv2.imwrite(name1, imMode1.Array)
            cv2.imwrite(name2, imMode2.Array)
            cv2.imwrite(name3, imMode3.Array)
            cv2.imwrite(name4, p1img)
            print('Save successfully')
        except Exception as e:
            print('Failed: ', str(e))
        cv2.waitKey(10)
        return BottleCount, imMode1, con1, p1img, p1data, imMode2, con2, p2img, p2data, imMode3
    except:
        print('Cannot processed')
        try:
            pathlib.Path(path + '\\Mode1\\').mkdir(parents=True, exist_ok=True)
            pathlib.Path(path + '\\Mode2\\').mkdir(parents=True, exist_ok=True)
            pathlib.Path(path + '\\Mode3\\').mkdir(parents=True, exist_ok=True)
            pathlib.Path(path + '\\barcode\\').mkdir(parents=True, exist_ok=True)

            name1 = os.path.join(path, 'Mode1', str(BottleCount) + '.tiff')
            name2 = os.path.join(path, 'Mode2', str(BottleCount) + '.tiff')
            name3 = os.path.join(path, 'Mode3', str(BottleCount) + '.tiff')
            print('Saving images...')
            cv2.imwrite(name1, imMode1.Array)
            cv2.imwrite(name2, imMode2.Array)
            cv2.imwrite(name3, imMode3.Array)
            print('Save successfully')
        except Exception as e:
            print('Failed: ', str(e))
        cv2.waitKey(10)
        return BottleCount, imMode1, 'FalseBC', None, 'Object not found', imMode2, 'FalseWL', None, 'Object not found', imMode3


def ExportCSV(data):
    df = pd.DataFrame(data)
    df.to_csv(path + str(datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')) + '.csv')
    print('CSV data has been exported')

def nothing(_):
    pass

if __name__ == '__main__':
    Window1 = pylon.PylonImageWindow()
    Window1.Create(1)
    Window2 = pylon.PylonImageWindow()
    Window2.Create(2)
    Window3 = pylon.PylonImageWindow()
    Window3.Create(3)
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    ser1 = fst.arduinoPort()
    st = fst.SaveFile_read()
    fst.ledcontrol_send(fst.Command_Input(st))
    print('Ready to work')
    path = r'G:\Shared drives\Optic\Common\Bottle_Project\Images\\' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H.%M'))

    Switch = np.zeros((300, 400, 3), np.uint8)
    cv2.namedWindow('Switch')
    switch = 'OFF \nON'
    cv2.createTrackbar(switch, 'Switch', 0, 1, nothing)
    data = {'Barcode':[],'Water level':[]}

    while True:
        cv2.imshow('Switch', Switch)
        k = cv2.waitKey(1) & 0xFF
        if s :
            s = cv2.getTrackbarPos(switch, 'Switch')
            condition = 'Start'
            ser1.write('Start\n'.encode())
            BottleCount, imMode1, con1, p1img, p1data, imMode2, con2, p2img, p2data, imMode3 = FirstProcess(st, ser1, path,  camera)
            print(BottleCount)
            if (BottleCount % 2) == 0:
                ser1.write(BottleCount)
            Window1.SetImage(imMode1)
            Window1.Show()
            Window2.SetImage(imMode2)
            Window2.Show()
            Window3.SetImage(imMode3)
            Window3.Show()
            data['Barcode'].append(p1data)
            data['Water level'].append(p2data)
        else:
            condition = 'Stop'
            ser1.write('Stop\n'.encode())
    cv2.destroyAllWindows()
    cv2.waitKey(1)


    ExportCSV(data)
    fst.ledcontrol_send(['cl'])
    print('Program has stopped')
