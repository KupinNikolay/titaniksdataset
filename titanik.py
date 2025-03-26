import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Загрузка данных
@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    return pd.read_csv(url)


df = load_data()

# Заголовок приложения
st.title("Анализ пассажиров Титаника")
st.markdown("Дашбоард для исследования датасета пассажиров Титаника")

# Раздел с описательной статистикой
st.header("1. Описательная статистика данных")
st.subheader("Первые 5 строк датасета")
st.write(df.head())

st.subheader("Информация о столбцах")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.subheader("Основные статистики")
st.write(df.describe())

st.subheader("Размер датасета")
st.write(f"Количество строк: {df.shape[0]}, Количество столбцов: {df.shape[1]}")

# Раздел с графиками
st.header("2. Визуализации данных")

# График 1: Распределение возрастов
st.subheader("Распределение возрастов пассажиров")
fig1, ax1 = plt.subplots()
sns.histplot(df["Age"].dropna(), kde=True, ax=ax1)
ax1.set_xlabel("Возраст")
ax1.set_ylabel("Количество")
st.pyplot(fig1)

# График 2: Выживаемость по классам
st.subheader("Выживаемость по классам билетов")
fig2, ax2 = plt.subplots()
sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax2)
ax2.set_xlabel("Класс билета")
ax2.set_ylabel("Количество")
ax2.legend(["Погиб", "Выжил"])
st.pyplot(fig2)

# График 3: Соотношение мужчин и женщин
st.subheader("Соотношение полов среди пассажиров")
fig3, ax3 = plt.subplots()
df["Sex"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax3)
ax3.set_ylabel("")
st.pyplot(fig3)

# График 4: Стоимость билетов по классам
st.subheader("Распределение стоимости билетов по классам")
fig4, ax4 = plt.subplots()
sns.boxplot(x="Pclass", y="Fare", data=df, ax=ax4)
ax4.set_xlabel("Класс билета")
ax4.set_ylabel("Стоимость")
st.pyplot(fig4)

# График 5: Выживаемость в зависимости от возраста и класса (реагирует на ввод)
st.header("3. Интерактивная визуализация")
st.subheader("Выживаемость в зависимости от возраста и класса билета")

# Пользовательский ввод
min_age, max_age = st.slider(
    "Выберите диапазон возрастов:",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(0, 80),
)

# Фильтрация данных по возрасту
filtered_df = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]

# Создание графика
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.violinplot(
    x="Pclass", y="Age", hue="Survived", data=filtered_df, split=True, ax=ax5
)
ax5.set_title(f"Выживаемость по возрасту и классу (возраст {min_age}-{max_age})")
ax5.set_xlabel("Класс билета")
ax5.set_ylabel("Возраст")
ax5.legend(title="Выжил", labels=["Нет", "Да"])
st.pyplot(fig5)

# Раздел с выводом строк
st.header("4. Просмотр данных")
n_rows = st.number_input(
    "Введите количество строк для отображения:",
    min_value=1,
    max_value=len(df),
    value=10,
)

st.write(f"Первые {n_rows} строк датасета:")
st.write(df.head(n_rows))

# Дополнительная информация
st.sidebar.header("О дашбоарде")
st.sidebar.info(
    "Этот дашбоард создан с помощью Streamlit для анализа датасета пассажиров Титаника. "
    "Он включает описательную статистику, визуализации и интерактивные элементы."
)
