# Generated by Django 3.0.3 on 2020-04-20 20:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sentiment', '0010_word2vecmodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='column',
            name='name',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='label',
            name='model_name',
            field=models.CharField(max_length=100, unique=True),
        ),
    ]
