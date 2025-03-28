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

# Создание возрастных групп
def create_age_groups(age):
    if pd.isnull(age):
        return 'Неизвестно'
    elif age < 18:
        return 'Дети (<18)'
    elif 18 <= age < 30:
        return 'Молодые (18-29)'
    elif 30 <= age < 50:
        return 'Взрослые (30-49)'
    else:
        return 'Пожилые (50+)'

df['AgeGroup'] = df['Age'].apply(create_age_groups)

# Функция для отображения информации о столбцах
def display_column_info(df):
    col_info = pd.DataFrame({
        'Столбец': df.columns,
        'Тип данных': df.dtypes.values,
        'Уникальных значений': df.nunique().values,
        'Пропусков': df.isnull().sum().values,
        '% пропусков': (df.isnull().mean() * 100).round(2).values
    })
    return col_info

# Интерфейс приложения
st.title("Анализ пассажиров Титаника")
st.markdown("### Интерактивный дашбоард")

# Раздел с описательной статистикой
st.header("1. Описательная статистика данных")

# Информация о столбцах
st.subheader("Подробная информация о столбцах")
st.dataframe(display_column_info(df))

# Расширенная статистика
st.subheader("Расширенная статистика для числовых столбцов")
st.write(df.describe(include='all').T)

# Визуализации
st.header("2. Визуальный анализ")

# Круговая диаграмма возрастных групп
st.subheader("Соотношение возрастных групп")
age_counts = df['AgeGroup'].value_counts()

fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
ax_pie.pie(age_counts, 
           labels=age_counts.index, 
           autopct='%1.1f%%',
           startangle=90,
           colors=sns.color_palette('pastel'),
           wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
ax_pie.set_title("Распределение пассажиров по возрастным группам", pad=20)
ax_pie.axis('equal')  # Чтобы диаграмма была круглой
st.pyplot(fig_pie)

# График 1: Распределение возрастов
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax1)
ax1.set_title("Распределение возраста пассажиров", fontsize=14)
ax1.set_xlabel("Возраст", fontsize=12)
ax1.set_ylabel("Количество", fontsize=12)
st.pyplot(fig1)

# График 2: Выживаемость по полу
st.subheader("Соотношение выживших по полу")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(x='Sex', hue='Survived', data=df, ax=ax2, palette='viridis')
ax2.set_title("Количество выживших/погибших по полу", fontsize=14)
ax2.set_xlabel("Пол", fontsize=12)
ax2.set_ylabel("Количество", fontsize=12)
ax2.legend(["Погиб", "Выжил"], title='Статус')
st.pyplot(fig2)

# График 3: Стоимость билетов по классам
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax3, palette='coolwarm')
ax3.set_title("Распределение стоимости билетов по классам", fontsize=14)
ax3.set_xlabel("Класс", fontsize=12)
ax3.set_ylabel("Стоимость билета", fontsize=12)
st.pyplot(fig3)

# Интерактивный элемент
st.header("3. Интерактивный анализ")
selected_class = st.selectbox(
    "Выберите класс каюты:",
    [1, 2, 3],
    index=0
)

filtered_df = df[df['Pclass'] == selected_class]
st.write(f"### Статистика для {selected_class} класса:")

# График 4: Возрастное распределение для выбранного класса
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.histplot(filtered_df['Age'].dropna(), bins=20, kde=True, ax=ax4)
ax4.set_title(f"Распределение возраста (класс {selected_class})", fontsize=14)
ax4.set_xlabel("Возраст", fontsize=12)
ax4.set_ylabel("Количество", fontsize=12)
st.pyplot(fig4)

# Вывод данных
st.header("4. Просмотр данных")
rows_to_show = st.slider(
    "Количество строк для отображения:",
    min_value=5,
    max_value=50,
    value=10
)
st.dataframe(df.head(rows_to_show))
