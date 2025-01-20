from django.core.exceptions import ValidationError
from django import forms


# Валідатор для додатних чисел і нуля
def validate_positive_or_zero(value):
    if value < 0:
        raise ValidationError("Значення має бути нулем або додатним числом.")


class LoanPredictionForm(forms.Form):
    """ Збір даних, необхідних для прогнозу схвалення кредиту """
    self_employed = forms.ChoiceField(choices=[(1, "Так"), (0, "Ні")], label="Самозайнятість")
    applicant_income = forms.FloatField(
        label="Доходи заявника",
        validators=[validate_positive_or_zero],
    )
    coapplicant_income = forms.FloatField(
        label="Доходи співзаявника",
        validators=[validate_positive_or_zero],
    )
    loan_amount = forms.FloatField(
        label="Сума кредиту",
        validators=[validate_positive_or_zero],
    )
    loan_term = forms.IntegerField(
        label="Термін кредиту (в місяцях)",
        validators=[validate_positive_or_zero],
    )
    credit_history = forms.ChoiceField(
        choices=[(1, "Хороша"), (0, "Погана")], label="Кредитна історія"
    )
    property_area = forms.ChoiceField(
        choices=[(1, "Міський"), (2, "Приміський"), (3, "Сільський")], label="Тип майна"
    )
    dependents = forms.ChoiceField(
        choices=[(0, "0"), (1, "1"), (2, "2"), (3, "3+")], label="Кількість залежних"
    )



