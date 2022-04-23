from django.shortcuts import render
# from silk.profiling.profiler import silk_profile


def index(request):
	return render(request, 'index.html')



