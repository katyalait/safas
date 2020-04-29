from data_handler import models
from rest_framework import serializers

class WordSerializer(serializers.ModelSerializer):

    class Meta:
        model = models.Word
        fields = ['word', 'date_added', 'categories']
