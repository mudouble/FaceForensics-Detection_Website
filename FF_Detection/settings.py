# from silk.profiling.profiler import silk_profile
import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # BASE_DIR就是manage.py文件的所在路径．

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "!)olkedj+0z_mydbsbc3gi(o1idjysl#159at!#y63$9mhikj*"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["0.0.0.0", "172.29.103.239", "127.0.0.1"]

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin", "django.contrib.auth",
    "django.contrib.contenttypes", "django.contrib.sessions",
    "django.contrib.messages", "django.contrib.staticfiles",
    "signup.apps.SignupConfig", "videos",
    "silk"
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "silk.middleware.SilkyMiddleware",
    # "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "FF_Detection.urls"

TEMPLATES = [{
    "BACKEND":
    "django.template.backends.django.DjangoTemplates",
    "DIRS": [
        os.path.join(BASE_DIR, "templates"),
        os.path.join(os.path.dirname(__file__), 'static'),
    ],
    "APP_DIRS":
    True,
    "OPTIONS": {
        "context_processors": [
            "django.template.context_processors.debug",
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
            "django.template.context_processors.media",
        ]
    },
}]

WSGI_APPLICATION = "FF_Detection.wsgi.application"

# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(BASE_DIR, "db.sqlite3"),
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME":
        "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {
        "NAME":
        "django.contrib.auth.password_validation.MinimumLengthValidator"
    },
    {
        "NAME":
        "django.contrib.auth.password_validation.CommonPasswordValidator"
    },
    {
        "NAME":
        "django.contrib.auth.password_validation.NumericPasswordValidator"
    },
]

# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Shanghai"

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = "/static/"

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
    # os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "preprocess_images"),
]

# STATIC_ROOT = os.path.join(BASE_DIR, "static")
LOGIN_REDIRECT_URL = "home"

LOGOUT_REDIRECT_URL = "home"

MEDIA_URL = "/media/"

MEDIA_ROOT = os.path.join(BASE_DIR, "media")
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
# 使用Python的内置cProfile分析器
SILKY_PYTHON_PROFILER = True

# 生成.prof文件，silk产生的程序跟踪记录，详细记录来执行来哪个文件，哪一行，用了多少时间等信息
SILKY_PYTHON_PROFILER_BINARY = True

# .prof文件保存路径
SILKY_PYTHON_PROFILER_RESULT_PATH = '/data/profiles/'


# @silk_profile(name='user login')  # name在Profiling页面区分不同请求名称
# def test(request):
#     pass


