import plotly.express as px
import pandas as pd
import pickle
from django.shortcuts import render
from .forms import LoanPredictionForm

# Словник для перекладу назв ознак
translation_dict = {
    'Self_Employed': 'Самозайнятість',
    'ApplicantIncome': 'Доходи заявника',
    'CoapplicantIncome': 'Доходи спільного заявника',
    'LoanAmount': 'Сума кредиту',
    'Loan_Amount_Term': 'Термін кредиту',
    'Credit_History': 'Кредитна історія',
    'Property_Area': 'Область нерухомості',
    'Dependents': 'Члени родини'
}

# Завантаження моделі
with open('loan_app/utils/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Завантаження моделі 1
with open('loan_app/utils/best_model1.pkl', 'rb') as file:
    best_model2 = pickle.load(file)

# Отримання важливості ознак
importances = best_model2.named_steps['classifier'].feature_importances_

# Отримуємо список назв ознак після трансформацій
numeric_feature_names = best_model2.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
categorical_feature_names = best_model2.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()

# Комбінуємо всі ознаки
feature_names = numeric_feature_names.tolist() + categorical_feature_names.tolist()

# Створення DataFrame для відображення важливості
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Переклад назви ознак на українську
importance_df['Feature'] = importance_df['Feature'].map(translation_dict).fillna(importance_df['Feature'])

# Замінюємо значення для "Самозайнятість" (для one-hot кодування)
importance_df['Feature'] = importance_df['Feature'].replace({
    'x0_No': 'Самозайнятість: Ні',
    'x0_Yes': 'Самозайнятість: Так'
})

# Сортуємо ознаки за важливістю
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Створення графіку важливості ознак
fig = px.bar(importance_df,
             x='Feature',
             y='Importance',
             title="Важливість ознак моделі",
             labels={'Importance': 'Важливість', 'Feature': 'Ознака'})

# Оновлення заголовків осей
fig.update_layout(
    title="Важливість ознак моделі",
    xaxis_title='Ознака',
    yaxis_title='Важливість'
)

# Перетворення графіку в HTML
graph_html = fig.to_html(full_html=False)

# Створення кругової діаграми для важливості ознак
fig_pie = px.pie(
    importance_df,
    names='Feature',
    values='Importance',
    title="Розподіл важливості ознак",
    labels={'Importance': 'Важливість', 'Feature': 'Ознака'}
)

# Перетворення графіку в HTML
graph_pie_html = fig_pie.to_html(full_html=False)


def loan_prediction(request):
    """ Обробка запиту для прогнозу схвалення кредиту на основі даних із форми користувача """
    result = None
    if request.method == "POST":
        form = LoanPredictionForm(request.POST)
        if form.is_valid():
            # Отримання даних із форми
            data = {
                'Self_Employed': [int(form.cleaned_data['self_employed'])],
                'ApplicantIncome': [form.cleaned_data['applicant_income']],
                'CoapplicantIncome': [form.cleaned_data['coapplicant_income']],
                'LoanAmount': [form.cleaned_data['loan_amount']],
                'Loan_Amount_Term': [form.cleaned_data['loan_term']],
                'Credit_History': [int(form.cleaned_data['credit_history'])],
                'Property_Area': [int(form.cleaned_data['property_area'])],
                'Dependents': [form.cleaned_data['dependents']],
            }
            input_data = pd.DataFrame(data)

            # Прогнозування
            prediction = best_model.predict(input_data)[0]
            result = "Ваш кредит схвалено!" if prediction == "Y" else "На жаль, ваш кредит не схвалено."
    else:
        form = LoanPredictionForm()

    return render(request, 'loan_app/predict.html', {'form': form, 'result': result})


def feature_importance(request):
    """ Відображення графіків важливості ознак та кругової діаграми """
    return render(request, 'loan_app/analysis_graphs.html', {
        'graph_html': graph_html,
        'graph_pie_html': graph_pie_html
    })




