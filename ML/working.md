## Importing System Libraries

## Preparing Image Data
```
BATCH_SIZE = 32
```
This is defining the size of the batch that will be used when training the model. In the context of machine learning, a batch is a subset of the dataset that the model is trained on before the weights are updated.
The batch size can significantly impact the performance and efficiency of the model training process.
```
IMAGE_SIZE = (224, 224)
```
This is defining the size that images will be resized to before they are input to the model. In many machine learning tasks involving images, it's necessary to resize all images to the same size so they can be input to the model.
The size (224, 224) is commonly used with certain types of models like CNNs because it's a balance between having enough detail for the model to learn from and not having so much detail that it becomes computationally expensive.

## Setting Dataset Path

```
dataset_path = r'C:\Users\Downloads\locust_pics'
```
The path to the dataset(`locust_pics`) is stored in `dataset_path` as a string.
```
`image_dir = Path(dataset_path)`
```
The `image_dir` is the `Path` object from the dataset path. This is done as `pathlib` module provide more functionality than simple strings for dealing with filesystem paths.
For example, in code we use the  `glob` method of the `Path` object to find all image files in the directory. This method is not available with a simple string.

So, `dataset_path` is used to store the raw path as a string, and `image_dir` is used when you need to perform operations on the path.
## Getting Filepaths
```
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpeg')) + list(image_dir.glob(r'**/*.png'))
```
The `glob` method is called on the `image_dir` object (which is assumed to be a Path object representing a directory). The `glob` method is used to find all the paths which match a certain pattern.

The pattern `**/*.JPG` is used to find all `.JPG` files in the directory represented by `image_dir` and its subdirectories. The `**` part of the pattern is a wildcard that matches any files or directories at any level of the directory tree. The `/*.JPG` part of the pattern matches any `.JPG` files.

The same process is repeated for `.jpeg` and `.png` files. The results of these three `glob` calls are then concatenated together using the + operator to form a single list of file paths, which is stored in the `filepaths` variable.

## Checking Filepaths

```
# Check if any files were found
if not filepaths:
    print(f"No image files found in directory {image_dir}")
```
This line checks if the filepaths list is empty. If it is, this means that no image files were found in the directory specified by image_dir.

```
else:
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
```
If the filepaths list is not empty (meaning that at least one image file was found), the code then proceeds to generate labels for these files.It does this by applying a function to each file path in the `filepaths` list using the map function.

The function being applied is a lambda function that takes a file path as input and returns the name of the parent directory of the file. This is done by calling `os.path.split(x)[0]` twice to get the parent directory of the file, and then `os.path.split(x)[1]` to get the name of this directory. The result of the map function is then converted to a list and stored in the `labels` variable.

This assumes that the images are stored in a directory structure where the name of the parent directory of each image file is its label.

## Convert filepaths and labels to pandas Series

```
filepaths_series = pd.Series(filepaths, name='Filepath')
labels_series = pd.Series(labels, name='Label')
```
The `pd.Series` function is used to convert the `filepaths` and `labels` lists into pandas `Series` objects. A pandas `Series` is a one-dimensional labeled array capable of holding any data type.

The `name` parameter is used to assign a name to the `Series` object, which will be used as the column name if the `Series` is later converted into a DataFrame.

The `filepaths_series` variable now holds a `Series` object with the file paths of the images, and the `labels_series` variable holds a `Series` object with the corresponding labels.

## Concatenate filepaths and labels

```
image_df = pd.concat([filepaths_series, labels_series], axis=1)
```
The `pd.concat` function is used to concatenate the `filepaths_series` and `labels_series` pandas `Series` objects along `axis=1`, which means they are concatenated column-wise (i.e., side by side).

The result is a DataFrame where each row corresponds to an image, with one column for the file path and one column for the label.

This DataFrame is stored in the `image_df` variable.

