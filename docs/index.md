# {{ Model Name (one line) }}

{{ Model Description (paragraph) }}

![{{model_id}}](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/odk_sample_agricultural_dataset.png)

MLHub model id: `{{model_id}}`. Browse on [Radiant MLHub](https://mlhub.earth/model/{{model_id}}).

## Training Data

{{

Provide links to the training data for this model. There should be separate
links for source and labels collections as the following example. Make sure to
include `Source` and `Labels` in the corresponding names of each collection.


Example using MLHub training data:

- [Training Data Source](https://api.radiant.earth/mlhub/v1/collections/ref_african_crops_kenya_02_source)
- [Training Data Labels](https://api.radiant.earth/mlhub/v1/collections/ref_african_crops_kenya_02_labels)

}}

## Related MLHub Dataset {{ (Optional) }}

{{

If this model was based on a dataset which is already published to MLHub, enter that link here.

[https://mlhub.earth/data/ref_african_crops_kenya_02](https://mlhub.earth/data/ref_african_crops_kenya_02)

}}

## Citation

{{

example:

Amer, K. (2022) “A Spatio-Temporal Deep Learning-Based Crop Classification
Model for Satellite Imagery”, Version 1.0, Radiant MLHub. [Date Accessed]
Radiant MLHub. <https://doi.org/10.34911/rdnt.h28fju>

}}

## License

{{

example: CC-BY-4.0

(update the LICENSE file in this repository to match the license)

}}

## Creator{{s}}

{{

example: Model creators and links go here (examples: Radiant Earth Foundation, Microsoft
AI for Good Research Lab).

}}

## Contact

{{

Contact email goes here (example: ml@radiant.earth)

}}

## Applicable Spatial Extent

{{

Here please provide the applicable spatial extent, for new inferencing (this
may be the same, or different than the spatial extent of the training data).
Please provide the spatial extent bounding box as WKT text or GEOJSON text.

```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "ID": 0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
              [-90,35],
              [-90,30],
              [-85,30],
              [-85,35],
              [-90,35]
          ]
        ]
      }
    }
  ]
}
```

<https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams#creating-geojson-and-topojson-maps>

}}

## Applicable Temporal Extent

{{

The recommended start/end date of imagery for new inferencing. Example:

| Start | End |
|-------|-----|
| 2000-01-01 | present |

}}

## Learning Approach

{{

The learning approach used to train the model. It is recommended that you use
one of the values below, but other values are allowed.

- Supervised
- Unsupervised
- Semi-supervised
- Reinforcement-learning
- Other (explain)

}}

## Prediction Type

{{

The type of prediction that the model makes. It is recommended that you use one
of the values below, but other values are allowed.

- Object-detection
- Classification
- Segmentation
- Regression
- Other (explain)

}}

## Model Architecture

{{

Identifies the architecture employed by the model. This may include any string
identifiers, but publishers are encouraged to use well-known identifiers
whenever possible. More details than just “it’s a CNN”!

}}

## Training Operating System

{{

Identifies the operating system on which the model was trained.

- Linux
- Windows (win32)
- Windows (cygwin)
- MacOS (darwin)
- Other (explain)

}}

## Training Processor Type

{{

The type of processor used during training. Must be one of "cpu" or "gpu".

- cpu
- gpu

}}

## Model Inferencing

Review the [GitHub repository README](../README.md) to get started running
this model for new inferencing.

## Methodology

{{

Use this section to provide more information to the reader about the model. Be
as descriptive as possible. The suggested sub-sections are as following:

}}

### Training

{{

Explain training steps such as augmentations and preprocessing used on image
before training.

}}

### Model

{{

Explain the model and why you chose the model in this section. A graphical representation
of the model architecture could be helpful to individuals or organizations who would
wish to replicate the workflow and reproduce the model results or to change the model
architecture and improve the results.

}}

### Structure of Output Data

{{

Explain output file names and formats, interpretation, classes, etc.

}}
