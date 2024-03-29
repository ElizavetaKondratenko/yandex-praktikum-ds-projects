# Прогнозирование оттока клиентов Банка
Код проекта - [ipynb][1]. Html версия - [html][2].

[1]: https://github.com/ElizavetaKondratenko/yandex-praktikum-ds-projects/blob/main/07-%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BE%D1%82%D1%82%D0%BE%D0%BA%D0%B0-%D0%BA%D0%BB%D0%B8%D0%B5%D0%BD%D1%82%D0%BE%D0%B2-%D0%B1%D0%B0%D0%BD%D0%BA%D0%B0/P7-bank-churn-prediction.ipynb
[2]: https://github.com/ElizavetaKondratenko/yandex-praktikum-ds-projects/blob/main/07-%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BE%D1%82%D1%82%D0%BE%D0%BA%D0%B0-%D0%BA%D0%BB%D0%B8%D0%B5%D0%BD%D1%82%D0%BE%D0%B2-%D0%B1%D0%B0%D0%BD%D0%BA%D0%B0/P7-bank-churn-prediction.html

## Основная задача

На основе данных из банка определить клиентов, которые могут уйти. 

## Описание проекта

**Заказчик** - «Бета-Банк». Необходимо провести анализ клиентов для создания системы предсказания, покинет ли клиент данный банк в ближайшее время или нет. Такая информация позволит подобрать подходящий способ удержать клиента в данном банке - по мнению банковских маркетологов сохранять текущих клиентов выйдет для банка дешевле, чем привлекать новых.

**Входные данные** - исторические данные о поведении клиентов и расторжении договоров с банком.

**Основная задача** - построить модель для задачи классификации, которая будет прогнозировать, уйдет ли клиент из банка в ближайшее время или нет. Необходимое условие: значение показателя F1-меры должно превышать порог в 0.59.

## Сферы деятельности

* Бизнес
* Инвестиции
* Банковская сфера/ ФинТех
* Кредитование
* Маркетинг

## Основные инструменты

- **python**
- **pandas**
- **numpy**
- **seaborn**
- **matplotlib**
- sklearn.preprocessing.**StandardScaler**
- sklearn.dummy.**DummyClassifier**
- sklearn.model_selection.**GridSearchCV**
- sklearn.linear_model.**LogisticRegression**
- sklearn.tree.**DecisionTreeClassifier**
- sklearn.ensemble.**RandomForestClassifier**


## Выводы

Нами была проделана работа по созданию модели, которая будет выявлять тех клиентов «Бета-Банка», которые могут в скором времени отказаться от услуг данного банка. Всего нами было рассмотрено три модели: логистическая регрессия, решающее дерево и случайный лес. Для каждой из них мы попытались подобрать наиболее оптимальные параметры, которые бы позволили достичь наибольшее значение F1-меры на тестовой выборке.

Моделирование проходило в два этапа. На первом этапе мы попытались построить модели, используя исходные данные с обнаруженным в них дисбалансом классов. Ни одна из рассмотренных нами моделей не смогла преодолеть порог ключевого показателя в 0.59. На втором этапе мы попытались учесть имеющийся дисбаланс: сменяли веса классов, увеличивали число наблюдений с положительными ответами. По итогу каждая из полученных на втором этапе моделей пересекла необходимый порог качества.

В качестве итоговой была выбрана модель случайного леса со следующими параметрами: max_depth=8 и n_estimators=50. Именно данная модель показала лучший результат на тренировочной выборке. В ней мы использовали увеличение количества положительных ответов для борьбы с дисбалансом.

После этого мы проверили качество нашей итоговой модели на тестовой выборке. К сожалению, качество нашей итоговой модели несколько упало в сровнение с результатом на тренировочных данных - такой исход может свидетельствовать о переобучении нашей итоговой модели.
