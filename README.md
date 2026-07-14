# proyecto-gnn-agrícola



Este proyecto se organizará como un pipeline reproducible para construir una base municipio-año destinada al modelado GNN espaciotemporal de soberanía alimentaria. La arquitectura separa claramente los datos originales, los datos intermedios, las bases limpias, la base maestra final y los archivos definitivos que serán consumidos por Python.



El flujo general será:



datos brutos

→ limpieza por fuente

→ validación y auditoría

→ feature engineering

→ integración municipio-año

→ exportación a Python

→ modelado GNN



La carpeta data/raw/ almacenará las bases originales sin modificar. Allí estarán las fuentes como EVA, DIVIPOLA, CHIRPS, ERA5-Land, CNA, UAF, precios agrícolas y otras bases auxiliares. Esta carpeta no debe editarse manualmente porque funciona como respaldo del dato original.



La carpeta data/interim/ guardará archivos temporales generados durante el procesamiento. Aquí pueden quedar bases como datos filtrados, versiones preliminares, cruces parciales o resultados usados solo como paso intermedio.



La carpeta data/processed/ contendrá las bases limpias por fuente. Por ejemplo, EVA limpia, variables climáticas agregadas por municipio-año, coordenadas municipales depuradas o bases socioeconómicas estandarizadas.



La carpeta data/final/ almacenará la base maestra integrada. El archivo principal será:



data/final/panel\_master.csv



Esta base reunirá variables productivas, climáticas, geográficas, socioeconómicas, territoriales y el target del modelo.



La carpeta más importante para el modelado será:



data/python\_ready/



Esta será la frontera entre R y Python. Todo archivo ubicado allí ya debe estar limpio, validado, integrado y listo para usar en modelos GNN.



Debe contener:



data/python\_ready/panel\_municipio\_anio.csv

data/python\_ready/features\_municipio\_anio.csv

data/python\_ready/target\_municipio\_anio.csv

data/python\_ready/coordenadas\_municipios.csv

data/python\_ready/splits\_temporales.csv



R se encargará principalmente de la limpieza, auditoría, integración y preparación de datos. Python se encargará de la construcción del grafo, generación de tensores, particiones temporales, entrenamiento y evaluación de modelos como GCN, GraphSAGE, GAT, T-GCN o modelos espaciotemporales.



La regla central de trabajo será:



R produce datos confiables

Python consume datos definitivos



Esto evita mezclar archivos experimentales, bases intermedias, reportes, auditorías y datasets finales. También reduce el riesgo de entrenar modelos con archivos incorrectos o contaminados.



La arquitectura inicial quedará así:



proyecto-gnn-agricola/

│

├── data/

│   ├── raw/

│   ├── interim/

│   ├── processed/

│   ├── final/

│   │   └── panel\_master.csv

│   └── python\_ready/

│       ├── panel\_municipio\_anio.csv

│       ├── features\_municipio\_anio.csv

│       ├── target\_municipio\_anio.csv

│       ├── coordenadas\_municipios.csv

│       └── splits\_temporales.csv

│

├── R/

│   ├── config/

│   ├── cleaning/

│   ├── features/

│   ├── audit/

│   ├── integration/

│   └── export/

│

├── python/

│   ├── graph/

│   ├── models/

│   ├── training/

│   └── evaluation/

│

└── outputs/

&#x20;   ├── audit/

&#x20;   ├── tables/

&#x20;   ├── figures/

&#x20;   └── reports/



En términos prácticos, el proyecto iniciará creando la estructura de carpetas y los archivos base de configuración:



R/config/00\_packages.R

R/config/01\_paths.R

R/config/02\_global\_parameters.R

R/config/export\_utils.R



Después se avanzará fuente por fuente, iniciando con EVA, porque es la base productiva principal. Luego se integrarán geografía municipal, clima, tierra, UAF, precios y variables complementarias.



El objetivo final no es producir muchas bases, sino una base confiable para modelado:



data/python\_ready/



Esa carpeta será la única que Python deberá leer para construir el grafo y entrenar los modelos.



