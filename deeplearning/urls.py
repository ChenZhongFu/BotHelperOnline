from django.conf.urls import url
import views
urlpatterns = [
    url(r'^predict', views.predict),
    url(r'^getModels', views.get_models),
    url(r'^deleteModel',views.delete_model),
]