# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2017-05-06 00:23
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='tweet',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('text', models.CharField(max_length=500)),
                ('preproc', models.CharField(max_length=500)),
                ('polarity', models.IntegerField()),
            ],
        ),
    ]
