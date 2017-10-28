from django.db import models

class tweet(models.Model):
    id = models.AutoField(serialize=True, primary_key=True)
    text=models.CharField(max_length=500,null=False)
    preproc=models.CharField(max_length=500,null=False)
    polarity=models.IntegerField(null=False)
    def __str__(self):
        return (str(self.id)+" | "+self.text+"|"+self.preproc+"|"+str(self.polarity))

