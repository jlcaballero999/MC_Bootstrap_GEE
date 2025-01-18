# MC_Bootstrap_GEE
# Simulaciones Monte Carlo en Google Earth Engine desde Google Colab

Este repositorio contiene un script en Python diseñado para ejecutar simulaciones de Monte Carlo utilizando Google Earth Engine (GEE) y realizar análisis estadísticos y visualizaciones de los resultados. 

## Requisitos

Asegúrate de contar con los siguientes elementos antes de ejecutar el código:

1. **Entorno de ejecución:**
   - Google Colab o un entorno local con acceso a Google Earth Engine y las bibliotecas necesarias.

2. **Credenciales de Google Earth Engine:**
   - Autentica y habilita tu cuenta de GEE para acceder a los datos y ejecutar las operaciones necesarias.

3. **Bibliotecas necesarias:**
   - Instala las dependencias:
     ```bash
     pip install geemap geopandas pandas matplotlib numpy
     ```

## Uso del código

### Importación de bibliotecas
El script utiliza las siguientes bibliotecas:

```python
import ee
import geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Configuración inicial
1. **Autenticación y configuración de GEE:**
   ```python
   ee.Authenticate()
   ee.Initialize(project='ee-alequech')
   ```

2. **Carga de datos:**
   - Colección de entrenamiento y capas de entrada:
     ```python
     training = ee.FeatureCollection('projects/ee-biomasa2/assets/Donw_Scaling/TrainingMU')
     PotFusionFA = ee.Image("projects/ee-biomasa2/assets/LangFA_gee_Clean_v3").rename("LangFA_gee")
     stacked = (
         ee.Image("projects/ee-fireforest2/assets/Landsat_30m_Amaz")
         .select(['VH_mean', 'VV_mean', 'elevation'])
         .addBands(PotFusionFA)
     )
     ```

3. **Conversión de datos:**
   - Convertir la colección de GEE a un DataFrame:
     ```python
     training_df = ee.data.computeFeatures({
         'expression': training,
         'fileFormat': 'GEOPANDAS_GEODATAFRAME'
     })
     training_df = training_df.set_crs('epsg:4326')
     ```

### Generación de muestras bootstrap
El código genera múltiples muestras bootstrap para realizar simulaciones Montecarlo:
```python
def create_bootstrap_samples(df, n_boot=10):
    samples = []
    original_size = len(df)
    for i in range(n_boot):
        sample_df = df.sample(n=original_size, replace=True, random_state=np.random.randint(0, 1e9))
        samples.append(sample_df)
    return samples

bootstrap_samples = create_bootstrap_samples(training_df, 100)
```

### Entrenamiento y clasificación
1. **Clasificación basada en muestras bootstrap:**
   ```python
   def classify_one_bootstrap(df, iteration):
       training_ee = geemap.geopandas_to_ee(df)
       model = (ee.Classifier.smileRandomForest(numberOfTrees=200)
                .setOutputMode('REGRESSION')
                .train(
                   features=training_ee,
                   classProperty='MU',
                   inputProperties=["MU", "LangFA_gee", "VH_mean", "VV_mean", "elevation"]
                ))
       classified_img = stacked.classify(model).rename(f'MU_{iteration}')
       return classified_img
   ```

2. **Creación de una colección de imágenes:**
   ```python
   image_list = []
   for i, df in enumerate(bootstrap_samples):
       classified_img = classify_one_bootstrap(df, i)
       image_list.append(classified_img)

   predictionCollection = ee.ImageCollection(image_list)
   ```

### Cálculo de métricas estadísticas
Se generan las métricas principales (promedio, desviación estándar y varianza):
```python
meanPrediction = predictionCollection.reduce(ee.Reducer.mean()).rename('MU_mean')
stdDevPrediction = predictionCollection.reduce(ee.Reducer.stdDev()).rename('MU_stdDev')
variancePrediction = predictionCollection.reduce(ee.Reducer.variance()).rename('MU_variance')
```

### Exportación de resultados
El script permite exportar los resultados a un asset de GEE:
```python
export_image = meanPrediction.addBands([stdDevPrediction, variancePrediction])

task = ee.batch.Export.image.toAsset(
    image=export_image,
    description='MonteCarlo_MU',
    assetId='projects/ee-alequech/assets/consultorias/peru/MonteCarlo_MU_100s_v2',
    region=stacked.geometry(),
    crs='EPSG:4326',
    scale=30,
    maxPixels=1e12
)
task.start()
```

### Verificación del estado de la tarea
Puedes monitorear el estado de la exportación:
```python
status = ee.data.getTaskStatus(task.id)[0]
print(status)
```

## Visualización de resultados
El script incluye visualización básica, como un histograma de la variable `MU`:
```python
training_df.plot.hist(column='MU', bins=30)
```

## Licencia
Este proyecto está licenciado bajo la [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
