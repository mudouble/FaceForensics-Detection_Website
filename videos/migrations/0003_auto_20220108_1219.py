# Generated by Django 2.2.12 on 2022-01-08 04:19

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0002_auto_20220107_2124'),
    ]

    operations = [
        migrations.CreateModel(
            name='Compressed_format',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='Forging_method',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
            ],
        ),
        migrations.RemoveField(
            model_name='videos_post',
            name='Compressed_format',
        ),
        migrations.RemoveField(
            model_name='videos_post',
            name='Forging_method',
        ),
        migrations.AddField(
            model_name='videos_post',
            name='compressed_format',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='videos.Compressed_format'),
        ),
        migrations.AddField(
            model_name='videos_post',
            name='forging_method',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='videos.Forging_method'),
        ),
    ]
