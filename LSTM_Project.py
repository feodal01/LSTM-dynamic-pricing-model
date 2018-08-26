# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
import matplotlib.pylab as plt
import time
import random
from sklearn.metrics import r2_score
from keras.models import load_model

def load_file(filename, sheet_name='', bin=True):
    # функция загружает эксель файл и делает его пандой
    if sheet_name != '':
        File = pd.read_excel(pd.ExcelFile(filename), sheet_name)
    else:
        File = pd.read_excel(pd.ExcelFile(filename), )
    File.index = File["Дата"]
    File = File.drop('Дата', axis=1)
    if bin == True:
        File = File.fillna(0)
    return File


def TrainScaler(Data,TargetData):
    # Обучаем скалер на загруженных данных
    scaler_exog = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_exog = scaler_exog.fit(Data)
    scaler_target = scaler_target.fit(pd.DataFrame(TargetData))
    # возвращает скалеры для данных и таргета
    return scaler_exog, scaler_target


def Reshape(Scaler,Data):
    # функция принимает на вход натренированный скалер и данные, которые надо преобразовать
    # с помощью скалера проводится нормализация, потом делается решейп
    # print(Data)
    NormalizedData = Scaler.transform(Data)
    NormalizedDataShaped = np.reshape(NormalizedData,
                                          (NormalizedData.shape[0], 1, NormalizedData.shape[1]))
    return NormalizedDataShaped


def TrainLSTModel(trainX,trainY):
    # формирование и тренировка модели
    EPOCHS = 30
    LAYER1 = 40
    ACTIVATOR = 'sigmoid'
    model = Sequential()
    model.add(LSTM(LAYER1, return_sequences=True, input_shape=(1, trainX.shape[2])))
    model.add(LSTM(20))
    model.add(Dense(1, activation=ACTIVATOR))
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    model.fit(trainX, trainY, epochs=EPOCHS,batch_size=1, verbose=2,
              )
    return model


def TrainAndTestModel(Separation):
    # Если разделитель  меньше 1, то идет разделение 70/30 и соответственно проводится тест на тестовой выборке
    # Загрузили сгенерированные данные
    target_name = 'Sales'
    Data = load_file('GeneratedData.xlsx',)
    #print(Data)
    TargetData = Data[target_name]
    GeneratedData = Data.drop(target_name, axis=1)
    print("Данные для тренировки модели загружены")
    time.sleep(1)
    if Separation<1:
        TRAIN_TEST = int(Separation*len(TargetData))  # 70% данных используется для обучения.
        print('Данные разбиты в пропорции 70/30 для обучения и теста модели')
        time.sleep(1)
        # Разбили данные на тест / трейн
        trainX = GeneratedData[:TRAIN_TEST]
        trainY = pd.DataFrame(TargetData[:TRAIN_TEST])
        testX = GeneratedData[TRAIN_TEST:]
        testY = TargetData[TRAIN_TEST:]
    else:
        print('Для обучения используется 100% данных без разбиения')
        trainX = GeneratedData
        trainY = pd.DataFrame(TargetData)

    # Обучили скалеры
    ScalerGeneratedData, ScalerTargetData = TrainScaler(trainX,trainY)
    # Нормализуем данные с помощью обученных скалеров
    NormalizedGeneratedData = ScalerGeneratedData.transform(trainX)
    NormalizedTargetData = ScalerTargetData.transform(trainY)
    # Сделали решейп нормализованных данных
    NormalizedReshapedGeneratedData = np.reshape(NormalizedGeneratedData,
                                              (NormalizedGeneratedData.shape[0], 1, NormalizedGeneratedData.shape[1]))
    NormalizedReshapedTargetData = np.reshape(NormalizedTargetData,
                                                 (NormalizedTargetData.shape[0], NormalizedTargetData.shape[1]))

    # Обучаем модель на основе нормализованных, решейпнутых данных
    print('Начинаем обучение модели на загруженных данных')
    time.sleep(1)
    LSTMModel = TrainLSTModel(NormalizedReshapedGeneratedData,NormalizedReshapedTargetData)
    LSTMModel.save('LSTModel.h5')
    print('Модель обучена. Файл модели сохранен на жесткий диск, файл LSTModel.h5')
    time.sleep(1)
    if Separation < 1:
        # готовим данные для теста модели(нормализуеи и делаем решейп)
        testX = Reshape(ScalerGeneratedData, testX)
    trainX = Reshape(ScalerGeneratedData, trainX)

    if Separation < 1:
        # формируем предсказание на основе обученной модели для тестовой выборки
        prediction = LSTMModel.predict(testX)
        inverted_prediction = pd.DataFrame(ScalerTargetData.inverse_transform(prediction))
    # формируем предсказание на основе обученной модели для тренировочной выборки
    prediction_train = LSTMModel.predict(trainX)
    inverted_prediction_train = pd.DataFrame(ScalerTargetData.inverse_transform(prediction_train))
    if Separation < 1:
        # считаем R2 для тренировочной и тестовой выборки для оценки качества предсказания
        print('R^2 на тренировочной выборке:', r2_score(pd.DataFrame(trainY), pd.DataFrame(inverted_prediction_train)))
        print('R^2 на тестовой выборке:', r2_score(pd.DataFrame(testY), pd.DataFrame(inverted_prediction)))
        # склеиваем предсказание для теста и тренировки и сбрасываем в эксель
        pred_full = inverted_prediction_train.append(inverted_prediction, ignore_index=True)
        pred_full = pred_full.set_index(TargetData.index)

        # Рисуем предсказание и реальные данные продаж
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set(title=str(str(target_name)), ylabel='Продажи')
        TargetData[:-1].plot(ax=ax, style='b', label='Observed')
        pred_full.plot(ax=ax,style='g--', label='Prediction')
        plt.show()

        pred_full.to_excel(pd.ExcelWriter('PredictionLSTM_70-30.xlsx'))
        print('Тест обученной модели проведен, предсказание модели сохранено '
              'на жесткий диск в файле PredictionLSTM_70-30.xlsx')

    else:
        pred_full = inverted_prediction_train
        pred_full.to_excel(pd.ExcelWriter('PredictionLSTM.xlsx'))
        print('Обучение модели на 100% данных произведено. '
              'Предсказание модели сохранено на жесткий диск в файле PredictionLSTM.xlsx')

    return LSTMModel, ScalerGeneratedData, ScalerTargetData


def FactorGenerator(Price_1kg):
    # Функция генерирует факторы на следующий шаг на основе последней строки в изначальных данных и заданной цены на 1нг

    # загружаем данные
    target_name = 'Sales'
    Data = load_file('GeneratedData.xlsx', )
    TargetData = Data[target_name].tail(3)
    GeneratedData = Data.drop(target_name, axis=1).tail(1)

    # рассчитываем факторы на следующий шаг
    Price_1kg_Substitute = float(GeneratedData['Price_1kg_Substitute'])
    CompetitorPrice = float(GeneratedData['CompetitorPrice'])
    Price = Price_1kg*10
    Ratio_Price_CometitorPrice = Price / CompetitorPrice
    Ratio_Price_SubstitudePrice1kg = Price_1kg / Price_1kg_Substitute
    Advert = int(GeneratedData['Акция'])
    Winter = int(GeneratedData['зима'])
    Spring = int(GeneratedData['весна'])
    Summer = int(GeneratedData['лето'])
    Autumn = int(GeneratedData['осень'])
    Shift1 = TargetData[-1]
    Shift2 = TargetData[-2]
    Shift3 = TargetData[-3]
    NewLine = [Advert, Winter, Spring, Summer, Autumn, Price_1kg_Substitute, Ratio_Price_SubstitudePrice1kg,
                     Ratio_Price_CometitorPrice, Shift1, Shift2, Shift3, Price, Price_1kg, CompetitorPrice]
    #print(NewLine)
    return NewLine

def GenerateSales(Data):
    # В данной функции происходит генерирование продаж на основе зависимостей
    # На вход подается строчка сгенерированных параметров
    # Формулы зависимостей взяты из файла Зависимости.xlsx

    Advert = random.uniform(0.5, 0.9)*Data[0]  # влиняие рекламной акции (величина влияния случайна от 0.5 до 0.9)
    Winter = 7*Data[1]  # Влияние зимы
    Spring = 54*Data[2]  # Влияние весны
    Summer = 95*Data[3]  # Влияние лета
    Autumn = 79*Data[4]  # Влияние осени
    # Влияние Ratio_Price_SubstitudePrice1k:
    Ratio_Price_SubstitudePrice1kg = (-19.951*Data[6]**2 + 83.492*Data[6]-68.952) * \
                                     sum((Winter, Spring, Summer, Autumn)) / 80
    # Влияние Ratio_Price_CometitorPrice
    Ratio_Price_CometitorPrice = (-74.735*Data[7]**2 + 148.34*Data[7] - 52.775) * sum((Winter, Spring, Summer, Autumn)) / 80
    Shift1 = (0.0011*Data[8]**2 -0.0631*Data[8] - 7.86) * sum((Winter, Spring, Summer, Autumn)) / 80 # Влияние сдвига продаж 1
    Shift2 = (-0.0373*Data[9] + 4.083) * sum((Winter, Spring, Summer, Autumn)) / 80 # Влияние сдвига продаж 2
    Shift3 = (-0.1498*Data[10] + 14.886) * sum((Winter, Spring, Summer, Autumn)) / 80 # Влияние сдвига продаж 3
    Price = (-9*10**(-5)*Data[11]**2 + 0.0756*Data[11] -14.197)  # Влияние цены
    WhiteNoise = random.randrange(-10,10)  # Рандом
    SumFactors = sum((Winter, Spring, Summer, Autumn, Ratio_Price_SubstitudePrice1kg, Ratio_Price_CometitorPrice,
                      Shift1, Shift2, Shift3, Price, WhiteNoise))

    SumFactors2 = sum((Winter, Spring, Summer, Autumn, WhiteNoise))

    if SumFactors < 0:  # Если продажи по сумме <0 тогда отталкиваясь от средних продаж за сезон используем рандом
        GeneratedSales = 0

    else: GeneratedSales = SumFactors # если продажи по сумме факторов больше 0, тогда берем как есть

    GeneratedSales = GeneratedSales * (1 + Advert)  # учитываем фактор рекламной акции
    return int(GeneratedSales)

def Costs(sales):
    # Функция считает издержки от заданных продаж
    # предполагается, что функция будет задаваться индивидуально под клиента
    # для демонстрационного случая функция примерно аналогична используемой в диссертации (без точных цифр)
    Data = load_file('GeneratedData.xlsx', )  # загружаем данные
    fixed_cost_per_item = Data['Price'].mean() * 0.7  # считаем 70% от средней цены - наши издержки
    return fixed_cost_per_item


def CalcRevenueAndProfit(price, sales):
    # функция считает выручку и прибыль на основе заданной цены и продаж
    revenue = price * sales
    profit = price * sales - sales * Costs(sales)
    return revenue, profit


def ChoosePrice(model, ScalerFactors, ScalerSales):
    # Функция осущесвляет перебор цен, предсказывая моделью продажи и выбирает цену с которой прибыль максимальна
    # загружаем данные, чтобы взять последнюю цену
    print('Цена может подбираться для максимизации прибыли, '
          'либо для максимизации выручки, в зависимости от целей ценообразования.')
    print('В демонстрационном случае цена подбирается для максимизации прибыли.')

    target_name = 'Sales'
    Data = load_file('GeneratedData.xlsx', )
    GeneratedData = Data.drop(target_name, axis=1).tail(1)
    Last_price = GeneratedData['Price_1kg']  # последняя цена на товар
    start_search = int(Last_price*0.5)
    end_search = int(Last_price*2)

    price = int(Last_price)*10  # базовая цена за упаковку 10кг
    sales = 0  # базовые продажи
    profit = 0  # базовый профит
    revenue = 0  # базовая выручка

    for prc in range(start_search, end_search, 1):
        # в цикле прогоняем каждую цену, предсказываем продажи на их основе
        TmpFactors = FactorGenerator(prc)  # генерируем строчку факторов на основе цены
        TmpFactors = Reshape(ScalerFactors, pd.DataFrame(TmpFactors).T)  # нормализуем и делаем решейп сген. строчки
        TmpPredictedSales = model.predict(TmpFactors)  # предсказываем продажи на основе готовой модели
        # инвертируем предсказание в читаемый вид:
        TmpInvertedPredictedSales = pd.DataFrame(ScalerSales.inverse_transform(TmpPredictedSales))
        # считаем ожидаемую выручку и прибыль:
        tmprevenue, tmpprofit = CalcRevenueAndProfit(prc*10, int(TmpInvertedPredictedSales[0]))

        # print(prc, tmprevenue, tmpprofit, int(TmpInvertedPredictedSales[0]))
        # если напредсказывали профит большем чем база, записываем:
        if tmpprofit > profit:
            price = prc
            sales = TmpInvertedPredictedSales
            profit = float(tmpprofit)
            revenue = tmprevenue

    print('Оптимизация завершена')
    time.sleep(1)
    print('Рекомендуемая цена за 1 кг: ', price)
    print('Ожидаемые продажи: ', int(sales[0]))
    print('Ожидаемая прибыль: ', int(profit))
    print('Ожидаемая выручка: ', int(revenue))

    return price, int(sales[0]), int(profit), int(revenue)

def client_input():
    # обрабатывает цену, введенную клиентом
    tmp2 = load_file('GeneratedData.xlsx', ).tail(1)['Price_1kg'][0]
    try:
        i = float(input())
        if i > 0:
            # проверяем, что введенная цена на слишком большая
            if i < tmp2 * 1000:
                return i
            else:
                print('Введенное вами число в ', round(i/tmp2,2), ' раз больше последней цены.')
                print('Пожалуйста, введите реалистичную цену:')
                return client_input()
        else:
            print('Пожалуйста, введите целое, положительное число')
            return client_input()
    except:
        print('Пожалуйста, введите целое, положительное число')
        return client_input()


def main_cycle_input():
    # обрабатывает ввод в главном меню
    try:
        tmp1 = int(input())
        if tmp1 in [1, 2, 3, 4]:
            return tmp1
        else:
            print("Введите число от одного до четырех")
            return main_cycle_input()
    except:
        print("Введите число от одного до четырех")
        return main_cycle_input()

def Demostration():
    print("========= Начало демонстрации =========")
    # обучаем модель на 100% данных
    LSTMModel, ScalerFactorsData, ScalerSalesData = TrainAndTestModel(1)
    time.sleep(1)

    print('Пожалуйста, введите цену на товар за 1кг для следующего периода. Товар продается в упаковке 10кг.')
    print('Продажи на основе введенной вами цены будут сравниваться с продажами на основе цены, выставленной моделью.')
    print('Это поможет сравнить эффективность выставления цены человеком и моделью.')
    tmp = load_file('GeneratedData.xlsx',)
    print('Цена за 1кг товара в прошлом периоде составила ', int(tmp.tail(1)['Price_1kg']))
    print('Цена за 10кг товара в прошлом периоде у конкурента составила ', int(tmp.tail(1)['CompetitorPrice']))
    print('Введите цену за 1 кг:')

    price_client = client_input()  # обрабатываем пользовательский ввод
    GeneratedFactorsData = FactorGenerator(price_client)  # генерируем список факторов на сл. период для клиентского ввода

    # рассчитывваем продажи на основе сгенерированных факторов по клиентской цене:
    SalesGeneratedBasedOnfactors = GenerateSales(GeneratedFactorsData)

    # считаем профит и выручку на основе цены клиента и сгенерированных продаж
    revenue_client, profit_client = CalcRevenueAndProfit(price_client*10, SalesGeneratedBasedOnfactors)

    print('Осуществляем подбор цены, на основе обученной модели в диапазоне -50% ... + 100% от последней цены.')
    # производим оптимизацию цены с помощью модели
    price_model, sales_model, profit_model, revenue_model = ChoosePrice(LSTMModel, ScalerFactorsData, ScalerSalesData)

    # генерируем факторы на основе цены выбранной моделью и продажи по этой цене
    GeneratedFactorsModel = FactorGenerator(price_model)
    SalesGeneratedModel = GenerateSales(GeneratedFactorsModel)

    # считаем выручку и профит на основе цены модели и сгенерированных продаж
    revenue_model_real, profit_model_real = CalcRevenueAndProfit(price_model*10, SalesGeneratedModel)

    Output = {'Показатели': ['Цена за 10кг', 'Продажи', 'Прибыль', 'Выручка'],
              'Клиент': [int(price_client)*10, int(SalesGeneratedBasedOnfactors), int(profit_client), int(revenue_client)],
              'Модель': [int(price_model)*10, int(SalesGeneratedModel), int(profit_model_real), int(revenue_model_real)],
              'Модель/Клиент, %': [[int(price_model/price_client*100)-100 if price_client != 0 else 'Н.п.'],
                                   [int(SalesGeneratedModel/SalesGeneratedBasedOnfactors*100)-100
                                    if SalesGeneratedBasedOnfactors != 0 else 'Н.п.'],
                                   [int(profit_model_real/profit_client*100)-100 if profit_client != 0 else 'Н.п.'],
                                   [int(revenue_model_real/revenue_client*100)-100 if revenue_client != 0 else 'Н.п.']]}
    OutputDataFrame = pd.DataFrame(data=Output).set_index('Показатели')
    print('')
    print('Итоговые результаты вычислены. Сравненительные результаты представлены ниже:')
    print('')
    print(OutputDataFrame)

    print('Демонстрация окончена. Нажмите любую клавишу')
    input()
    print("Чтобы показать информацию о предпосылках демонстрации, введите 1")
    print("Чтобы протеситровать предсказательную силу модели, введите 2")
    print("Чтобы начать демонстрацию, введите 3")
    print('Если у вас уже есть обученная модель и вы хотите оптимизировать цену на следующий период, введите 4')
    print('Для выхода закройте окно скрипта')


def main_sycle():
    print('Введите пункт меню 1, 2, 3 или 4')
    x = main_cycle_input()
    if x == 1:
        print('=============Предпосылки============')
        print('1) Издержки рассчитываются на основе функции вида: 80% от средней ццены товара '
              'за все предшествующие периоды.')
        print('Эта функция является аналогом "закупочная цена + фиксированная надбавка" и '
              'в действительности используется на практике.'
              'Функция можем быть легко изменена в завимисимости от потребностей.')
        print('')
        print('2) При демонстрации происходит расчет объема продаж на следующий период'
              ' на основе высталенной клиентом и моделью цены.'
              'Данный расчет основывается на восстановленной кривой спроса с реальных данных. '
              'Все формы зависимостей реальны.')
        print('Таким образом, расчет продаж является правдоподобным на столько, на сколько это возможно. '
              'При этом, расчет продаж не является копией продаж с базы практики. '
              'Таким образом, коммерческая тайна не нарушена.')
        print('')
        print('3) Список факторов в демонстрации является ограниченным. '
              'Он может быть полностью изменен в зависимости от потребности и условий ведения бизнеса.')
        main_sycle()

    if x == 2:
        print('======== Тест модели ========')
        LSTMModel, ScalerGeneratedData, ScalerTargetData = TrainAndTestModel(0.7)
        main_sycle()
    if x == 3:
        Demostration()
        main_sycle()
    if x ==4:
        print('======== Оптимизация ========')
        print('Загружаем обученную модель LSTModel.h5')
        model = load_model('LSTModel.h5')
        time.sleep(1)
        print('Модель загружена')
        print('Загружаю данные по предыдущим продажам. '
              'В оптимизации будут использованы факторы (Цена конкурента, сезон, акция и т.д.) из последней строки данных')

        # обучаем скалеры чтобы передать оптимизатору
        target_name = 'Sales'
        Data = load_file('GeneratedData.xlsx', )
        TargetData = pd.DataFrame(Data[target_name])
        GeneratedData = Data.drop(target_name, axis=1)
        ScalerGeneratedData, ScalerTargetData = TrainScaler(GeneratedData, TargetData)
        # проводим оптимизацию цены
        price_model, sales_model, profit_model, revenue_model = ChoosePrice(model, ScalerGeneratedData, ScalerTargetData)
        main_sycle()

print('Начало демонстрационного скрипта')
print("Чтобы показать информацию о предпосылках демонстрации, введите 1")
print("Чтобы протеситровать предсказательную силу модели, введите 2")
print("Чтобы начать демонстрацию, введите 3")
print('Если у вас уже есть обученная модель и вы хотите оптимизировать цену на следующий период, введите 4')
print('Для выхода закройте окно скрипта')
main_sycle()