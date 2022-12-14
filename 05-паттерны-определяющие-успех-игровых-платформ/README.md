# Паттерны, определяющие успех игровых платформ
Код проекта - [ipynb][1]. Html версия - [html][2].

[1]: https://github.com/ElizavetaKondratenko/yandex-praktikum-ds-projects/blob/main/05-%D0%BF%D0%B0%D1%82%D1%82%D0%B5%D1%80%D0%BD%D1%8B-%D0%BE%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D1%8F%D1%8E%D1%89%D0%B8%D0%B5-%D1%83%D1%81%D0%BF%D0%B5%D1%85-%D0%B8%D0%B3%D1%80%D0%BE%D0%B2%D1%8B%D1%85-%D0%BF%D0%BB%D0%B0%D1%82%D1%84%D0%BE%D1%80%D0%BC/P5-patterns-that-determine-the-success-of-game-platforms.ipynb
[2]: https://github.com/ElizavetaKondratenko/yandex-praktikum-ds-projects/blob/main/05-%D0%BF%D0%B0%D1%82%D1%82%D0%B5%D1%80%D0%BD%D1%8B-%D0%BE%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D1%8F%D1%8E%D1%89%D0%B8%D0%B5-%D1%83%D1%81%D0%BF%D0%B5%D1%85-%D0%B8%D0%B3%D1%80%D0%BE%D0%B2%D1%8B%D1%85-%D0%BF%D0%BB%D0%B0%D1%82%D1%84%D0%BE%D1%80%D0%BC/P5-patterns-that-determine-the-success-of-game-platforms.html

## Основная задача

На основе исторических данных о продажах компьютерных игр, рейтингах пользователей и экспертов, жанрах и платформах, выявить закономерности, определяющие успех игры. 

## Описание проекта

**Заказчик** — интернет-магазин «Стримчик», который продаёт по всему миру компьютерные игры. Необходимо выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.

**Входные данные** — собранные из открытых источников исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы. Вам нужно выявить определяющие успешность игры закономерности. Данные покрывают временной период с 1980 по 2016 года включительно.

## Сферы деятельности

* Gamedev
* Интернет-магазины
* Маркетинг

## Основные инструменты

- **python**
- **pandas**
- **numpy**
- **matplotlib**
- **scipy**

## Выводы

Итак, нами был проведен предварительный анализ выпускаемых видеоигр с целью выявления закономерностей, которые наиболее сильно влияют на успешность игры. Эту информацию поможет компании-заказчику сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.

Какие закономерности нам удалось выявить?
1. В последние несколько лет стремительно набирают популярность недавно вышедшие платформы PS4 и XOne. В ближайшее время игры будут выходить преимущественно на данные платформы. Среди компактных платформ выделяется Nintendo 3DS.
2. Оценки как критиков, так и пользователей имеют положительное влияние на объемы продаж, однако сила влияния первой группы сильнее.
3. Наиболее популярный и прибыльный жанр - шутер. Именно на него стоит сделать ставку. Также можно обратить внимание на ролевые игры, ведь нередко они выпускаются под двойным жанром Action/RPG.
4. В последние несколько лет наибольшей популярностью у пользователей из Америки и Европы пользуется последняя вышедшая консоль серии PlayStation. Японские пользователи также предпочитают данную серию консолей, но наибольшей популярностью у них пользуются именно компактные переносные консоли из серии Nintendo.
5. Американцы и европейцы отдают предпочтение взрослым играм, а на втором месту у них игры для всех возрастов. У японских пользователей, напротив, на первом месте располагаются игры для подростков, а на втором - игры для всех возрастов. Однако результаты для японских пользователей могут быть менее точными в силу существенного числа игр без рейтинга.
6. Средние оценки на платформах серии Xbox и ПК значимо не разнятся. Для играков куда важнее содержание игры, чем предназначенная для нее платформа.
7. А вот средние оценки игр жанров экшен и спорт существенно разнятся.
