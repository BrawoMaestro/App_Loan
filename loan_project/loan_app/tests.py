import pickle
from django.test import TestCase
from django.urls import reverse
import os

""" python manage.py test loan_app.tests """

# Замість 'loan_project.settings' вкажіть шлях до вашого файлу налаштувань
os.environ['DJANGO_SETTINGS_MODULE'] = 'loan_project.settings'


class FeatureImportanceViewTest(TestCase):
    def test_feature_importance_view(self):
        """Перевірка відображення графіків важливості ознак"""
        response = self.client.get(reverse('feature_importance'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Важливість ознак моделі')  # Перевірка наявності графіку
        self.assertContains(response, 'Розподіл важливості ознак')  # Перевірка наявності кругової діаграми


class ModelLoadingTest(TestCase):
    databases = {'default': 'sqlite://:memory:'}  # Вказуємо використання бази даних тільки для тестів

    def test_model_loading(self):
        """Перевірка завантаження моделей"""
        with open('loan_app/utils/best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        self.assertIsNotNone(model)

        with open('loan_app/utils/best_model1.pkl', 'rb') as file:
            model2 = pickle.load(file)
        self.assertIsNotNone(model2)


