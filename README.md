### spark-pca

#take  
copy iris.json to /tmp
run with spark-submit pca.py > pca.out
python plotpca.py

takes the iris dataset, reduces the 4 dimensional form of the sepal/petal length/widths and
produces a PCA result in 2 dimensions. This is then plotted in python/matplotlib in three forms
