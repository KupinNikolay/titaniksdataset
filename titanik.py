import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

# Интерфейс приложения
st.title("Анализ пассажиров Титаника")
st.markdown("### Интерактивный дашбоард")

# Описательная статистика
st.header("1. Описательная статистика")
st.subheader("Основная информация")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.subheader("Числовые характеристики")
st.write(df.describe())

# Визуализации
st.header("2. Визуальный анализ")

# График 1: Распределение возрастов
fig1, ax1 = plt.subplots()
sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax1)
ax1.set_title("Распределение возраста пассажиров")
st.pyplot(fig1)

# График 2: Выживаемость по полу (заменён на гистограмму)
st.subheader("Соотношение выживших по полу")
fig2, ax2 = plt.subplots()
sns.countplot(x='Sex', hue='Survived', data=df, ax=ax2)
ax2.set_title("Количество выживших/погибших по полу")
ax2.set_xlabel("Пол")
ax2.set_ylabel("Количество")
ax2.legend(["Погиб", "Выжил"])
st.pyplot(fig2)

# График 3: Стоимость билетов по классам
fig3, ax3 = plt.subplots()
sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax3)
ax3.set_title("Распределение стоимости билетов")
st.pyplot(fig3)

# Интерактивный элемент
st.header("3. Интерактивный анализ")
selected_class = st.selectbox(
    "Выберите класс каюты:",
    [1, 2, 3],
    index=0
)

filtered_df = df[df['Pclass'] == selected_class]
st.write(f"Данные для {selected_class} класса:")

# График 4: Возрастное распределение для выбранного класса
fig4, ax4 = plt.subplots()
sns.histplot(filtered_df['Age'].dropna(), bins=20, kde=True, ax=ax4)
ax4.set_title(f"Распределение возраста (класс {selected_class})")
st.pyplot(fig4)

# Вывод данных
st.header("4. Просмотр данных")
rows_to_show = st.slider(
    "Количество строк для отображения:",
    min_value=5,
    max_value=50,
    value=10
)
st.write(df.head(rows_to_show))
