# Generated by Django 3.0.3 on 2020-03-01 20:15

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data_handler', '0009_auto_20200301_1900'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='stockprice',
            options={'ordering': ['date']},
        ),
    ]
