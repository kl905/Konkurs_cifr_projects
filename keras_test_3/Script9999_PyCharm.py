import pyaudio
import wave
from PIL import Image
import numpy as np
import cv2
import os
import pytesseract
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
import tensorflow as tf

def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),  # Нормализация входных данных

            tf.keras.layers.Conv2D(64, (3, 3),padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3),padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),


            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(33, activation='softmax') # число выходных классов


        ]
    )
    return model


def rgb_of_pixel(img_path, ReadBrille):
    try:
        a = ""
        R = 500
        G = 500
        B = 500
        Letters = {"000000": "Полная тьма", "111111": "Ничего", "100000": 'А', "101000": 'Б', "011101": 'В',
                   "111100": 'Г', "110100": 'Д', "100100": 'Е', "100001": 'Ё', "011100": 'Ж', "100111": 'З',
                   "011000": 'И', "111011": 'Й', "100010": 'К', "101010": 'Л', "110010": 'М', "110110": 'Н',
                   "100110": 'О', "111010": 'П', "101110": 'Р', "011010": 'С', "011110": 'Т', "100011": 'У',
                   "111000": 'Ф', "101100": 'Х', "110000": 'Ц', "111110": 'Ч', "100101": 'Ш', "110011": 'Щ',
                   "101111": 'Ъ', "011011": 'Ы', "011111": 'Ь', "011001": 'Э', "101101": 'Ю', "111001": 'Я'}
        CoordsXL = [234, 349, 239, 346, 261, 352]
        CoordsYL = [45, 48, 157, 160, 307, 313]
        CoordsXR = [345, 447, 348, 446, 343, 440]
        CoordsYR = [153, 150, 291, 297, 402, 404]
        im = Image.open(img_path).convert('RGB')
        wid, hgt = im.size
        for i in range(len(CoordsXL)):
            x = 0
            y = 0
            R = 500
            G = 500
            B = 500
            sumr = 0
            sumg = 0
            sumb = 0
            pixels = abs(CoordsYL[i] - CoordsYR[i]) * abs(CoordsXL[i] - CoordsXR[i])
            while y != abs(CoordsYL[i] - CoordsYR[i]):
                while x != abs(CoordsXL[i] - CoordsXR[i]):
                    r, g, b = im.getpixel((CoordsXL[i] + x, CoordsYL[i] + y))
                    sumr = sumr + r
                    sumg = sumg + r
                    sumb = sumb + r
                    if r < R:
                        R = r
                    if g < G:
                        G = g
                    if b < B:
                        B = b
                    x += 1
                y += 1
                x = 0
            # r, g, b = im.getpixel((CoordsX[i]q, CoordsY[i]))
            print("RGB: ", R + G + B)
            print("average: ", (sumr + sumg + sumb) / (pixels))
            if R + G + B < (sumr + sumg + sumb) / (3 * pixels):
                found = 0
                blackPixels = 0
                x = -1
                y = -1
                # print("2")
                while y != abs(CoordsYL[i] - CoordsYR[i]):
                    y += 1
                    x = 0
                    if found == 1:
                        break
                    while x != abs(CoordsXL[i] - CoordsXR[i]):
                        x += 1
                        if found == 1:
                            break
                        r, g, b = im.getpixel((CoordsXL[i] + x, CoordsYL[i] + y))
                        x2 = -5
                        y2 = -5
                        # print(r+g+b,'',(sumr+sumg+sumb)/(3*pixels))
                        # print(x,'',y,'',abs(CoordsXL[i]-CoordsXR[i]),'',abs(CoordsYL[i]-CoordsYR[i]))
                        if r + g + b < (sumr + sumg + sumb) / (3 * pixels):
                            # print("1")
                            while y2 != 5:
                                if found == 1:
                                    break
                                while x2 != 5:
                                    r, g, b = im.getpixel((CoordsXL[i] + x + x2, CoordsYL[i] + y + y2))
                                    # print('1')
                                    if r + g + b < (sumr + sumg + sumb) / (3 * pixels):
                                        blackPixels += 1
                                        if blackPixels == 50:
                                            print("blackPixels:", blackPixels)
                                            a = a + '0'
                                            found = 1
                                            break
                                    x2 += 1
                                y2 += 1
                                x2 = 0
                if found == 0:
                    print(a)
                    a = a + '1'
            else:

                a = a + '1'
        if cv2.waitKey(1) == ord('z'):
            pass
        if cv2.waitKey(100) == ord('z'):
            path = r'Sounds\Режим-карточки' + '.wav'
            ReadBrille = 0
            try:
                audio_file = wave.open(path)
                FORMAT = audio_file.getsampwidth()  # глубина звука
                CHANNELS = audio_file.getnchannels()  # количество каналов
                RATE = audio_file.getframerate()  # частота дискретизации
                N_FRAMES = audio_file.getnframes()  # кол-во отсчетов
                audio = pyaudio.PyAudio()

                # открываем поток для записи на устройство вывода - динамик - с такими же параметрами
                out_stream = audio.open(format=audio.get_format_from_width(FORMAT),
                                        channels=CHANNELS, rate=RATE, output=True)

                out_stream.write(audio_file.readframes(N_FRAMES))  # отправляем на динамик

                audio.terminate()
            except:
                pass
            return "Режим карточки"
        return Letters[a]
    except:
        return "Не знаю"


Let_num=[]
t=0
ReadBrille=1
Letters={"а":0,"б":0,"в":0,"г":0,"д":0,"е":0,"ё":0,"з":0,"и":0,"й":0,"к":0,"л":0,"м":0,"н":0,"о":0,"п":0,"р":0,"с":0,"т":0,"у":0,"ф":0,"х":0,"ц":0,"ч":0,"ш":0,"щ":0,"ъ":0,"ы":0,"ь":0,"э":0,"ю":0,"я":0,"ж":0}
x = 0
stop=1
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('й'):
        break
    if cv2.waitKey(100) == ord(' '):
        stop=0
        x=0
    if cv2.waitKey(100) == ord('z'):
        if ReadBrille == 0:
            print("Режим кубик")
            path = r'Sounds\Режим-кубика' + '.wav'
            ReadBrille = 1
        else:
            print("Режим карточки")
            path = r'Sounds\Режим-карточки' + '.wav'
            ReadBrille = 0
        audio_file = wave.open(path)
        FORMAT = audio_file.getsampwidth()  # глубина звука
        CHANNELS = audio_file.getnchannels()  # количество каналов
        RATE = audio_file.getframerate()  # частота дискретизации
        N_FRAMES = audio_file.getnframes()  # кол-во отсчетов
        audio = pyaudio.PyAudio()

        # открываем поток для записи на устройство вывода - динамик - с такими же параметрами
        out_stream = audio.open(format=audio.get_format_from_width(FORMAT),
                                channels=CHANNELS, rate=RATE, output=True)

        out_stream.write(audio_file.readframes(N_FRAMES))  # отправляем на динамик

        audio.terminate()



    if ReadBrille==1 and stop!=1:
        image_path_cube = r'IMG_cube/Image.jpg'
        cv2.imwrite(image_path_cube, frame)
        brilleText = rgb_of_pixel(image_path_cube, ReadBrille)
        print(brilleText)
        if brilleText == "Режим карточки":
            ReadBrille = 0
        elif brilleText != " " and brilleText != "Ничего" and brilleText != "Полная тьма":
            print(brilleText.lower(),'asasasdsa')
            path = r'Sounds\Буква ' + brilleText.lower() + '.wav'
            audio_file = wave.open(path)
            FORMAT = audio_file.getsampwidth()  # глубина звука
            CHANNELS = audio_file.getnchannels()  # количество каналов
            RATE = audio_file.getframerate()  # частота дискретизации
            N_FRAMES = audio_file.getnframes()  # кол-во отсчетов
            audio = pyaudio.PyAudio()
            # открываем поток для записи на устройство вывода - динамик - с такими же параметрами
            out_stream = audio.open(format=audio.get_format_from_width(FORMAT),
                                    channels=CHANNELS, rate=RATE, output=True)

            out_stream.write(audio_file.readframes(N_FRAMES))  # отправляем на динамик

            audio.terminate()
            x+=1
            if x==2:
                stop = 1

    if ReadBrille==0 and stop!=1:
        image_path = r'DataSetPred/class_m/savedImage.jpg'
        frame_re = frame[0:480, 90:560]
        cv2.imwrite(image_path, frame_re)


        if t==0:
            batch_size = 33
            img_height = 350
            img_width = 470
            num_classes = 33

            # Create a new model instance
            model = create_model()

            # Restore the weights
            model.load_weights('checkpoints/my_checkpoint.h5')

            model.compile(
                optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
            t=1
        batch_size = 33
        img_height = 350
        img_width = 470
        predict_ds = tf.keras.utils.image_dataset_from_directory(
            'DataSetPred',
            color_mode='rgb',
            # скорость срабатывания
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        #Predict model
        Let_num=list(np.argmax(model.predict(predict_ds), axis=-1))

        test=[["а",0],["б",1],["в",2],["г",3],["д",4],["е",5],["ё",6],["ж",7],["з",8],["и",9],["й",10],["к",11],["л",12],["м",13],["н",14],["о",15],["п",16],["р",17],["с",18],["т",19],["у",20],["ф",21],["х",22],["ц",23],["ч",24],["ш",25],["щ",26],["ъ",27],["ы",28],["ь",29],["э",30],["ю",31],["я",32]]
        for i in range(len(test)):
            for j in range(len(Let_num)):
                if Let_num[j] in test[i]:
                    try:
                        # Настрока голоса
                        path=r'Sounds\Буква '+ test[i][0] +'.wav'
                        audio_file = wave.open(path)
                        FORMAT = audio_file.getsampwidth()  # глубина звука
                        CHANNELS = audio_file.getnchannels()  # количество каналов
                        RATE = audio_file.getframerate()  # частота дискретизации
                        N_FRAMES = audio_file.getnframes()  # кол-во отсчетов
                        audio = pyaudio.PyAudio()

                        # открываем поток для записи на устройство вывода - динамик - с такими же параметрами
                        out_stream = audio.open(format=audio.get_format_from_width(FORMAT),
                                                channels=CHANNELS, rate=RATE, output=True)

                        out_stream.write(audio_file.readframes(N_FRAMES))  # отправляем на динамик

                        audio.terminate()
                        x += 1
                        if x ==2:
                            stop=1
                    except:
                        print("Нет выхода для звука")
cap.release()
cv2.destroyAllWindows()
