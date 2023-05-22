## ML-Ops Project 2

### Classical Data Analysis

* Firstly, I added the necessary libraries.
* Then, I pulled the data assigned as homework from the GitHub Repo of the BootCamp instructor.
    * `df = pd.read_csv('https://github.com/erkansirin78/datasets/raw/master/housing.csv')`
    * Actually, this command is an API command.
* After that, I briefly did exploratory data analysis.
* I examined the correlation between variables, which are numerical and paired, excluding `ocean_proximity`.

* Then I visualized them with a scatterplot for paired relationships again.
* Evaluating the strength of their relationships here is important.
* Later, I evaluated the missing values. Here I will handle the missing value along with PyCaret.
* Then, because there is geographic information, I wanted to display it.
* With a good understanding of California's map, it becomes evident that the areas with high population density, such as the Bay Area, Los Angeles, and San Diego, are easily identifiable.
* Our brains have a natural ability to visually recognize patterns, but it is essential to manipulate visualizations to emphasize these patterns effectively.
* In the visualization by locations, I first set the sizes of the scatters with 'population' for a 4th dimension, and then colored it according to the `median_house_value`.
* Could I show this with a heat map too? Of course, I created a heat map according to the frequency of house listings using the folium library as I had encountered before on Kaggle.
* I visualized their numbers according to ocean_proximity.

### Ml - Ops

* Then let's move on to the MlOps part;
* As I explained in Week_1, I use MlFlow and MySQL with installations. These containers are actively running on Oracle infrastructure at the moment. I can connect constantly.
* We install the necessary environments here I used localhost but you can use your own ip address or private configurations.
* I didn't know PyCaret wanted such a config. You can read the medium I benefited from this notebook, [Moez Ali's the developer of PyCaret](https://moez-62905.medium.com/simplify-mlops-with-pycaret-mlflow-and-dagshub-366c768f0dac), and my inspiration. 
* The arguments we need to add in the standard installation are as follows;

```
log_experiment = True,
log_data=True,
#log_profile=True, # wasting time
log_plots=True,
system_log = True,
experiment_name='MLFlow_by_PyCaret'
```
* The important thing here is that if there is a PyCaret `.env` file, it takes it automatically, so if there are unnecessary keys in the `.env` file, you will get an error.
* You should make the `experiment_name` selection unique, it can change, or it says it stays active.

#### Lets start to experiments
* PyCaret now automatically throws all model trials into the experiment part of mlflow. It combines all the trials on the setup created with the first start, so even if it looks like a single page, you can see all the saved trials by pressing the + part. With a recent PyCaret update, html results could not be displayed in some idles. This made it even easier for me.
* For example, all the logs of all the models in compare_model() are automatically transferred. Not only these models, but also an interactive table where you can compare the results of all tried models will certainly be a great pleasure.
* All plots printed are again logged in the same way and can be pulled from your MySQL whenever you want and looked at as you wish.

#### Anyway, let's finish the model and save the best models;
* Now if you have chosen, you can record your best models, for this I created a simple iterative function

* But there's an obvious problem here;

* PyCaret automatically logs the performance metrics, hyperparameters, and other details of all models once they are directed to MLFlow. However, if you want to save the model itself, there is a different approach. In this approach, we first finalize the PyCaret model and then complete the process using the log_model function. In the template code, sklearn is used as an example, but there are also specific functions available for some ML libraries (such as PyTorch or FastAI). I don't know if PyCaret will have a specific function like this in the future. One challenge I haven't overcome yet is saving the models as versions. However, it is possible to load only one version of the models and use separate names for each. But I personally find it more elegant to save a trial process as versions of a model. When you enter the models, it already indicates which algorithm was used.

* After this process is completed, you can review your models in the Registered Model section.

#### What if we wanted to delete or list the registered models without opening in a browser;
* For this, I defined a usable function, it automatically takes the environment configurations.
* Please stay tuned for my Ml-Ops projects.