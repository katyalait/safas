# Generated by Django 3.0.3 on 2020-02-24 16:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data_handler', '0005_auto_20200224_1409'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='date_written',
            field=models.DateTimeField(default='2020-02-24 16:56:24'),
        ),
    ]
