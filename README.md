# Media Anonymization Plugin

This plugin is a Python plugin that allows you to anonymize media in your
FiftyOne dataset by running a zero-shot text search model to detect objects
in the images and apply a blur

1. Define search query like "human faces", "license plates", or even multiple terms at once like "car,bus,boat"
2. Applies OWL-ViT model to detect objects matching the search term.
3. Blur each region of interest and save the new image.
4. Saves to a new media field for anonymized FO app consumption!

## Installation

```shell
fiftyone plugins download https://github.com/swheaton/twilio-automation-plugin
```

Refer to the [main README](https://github.com/voxel51/fiftyone-plugins) for
more information about managing downloaded plugins and developing plugins
locally.

## Usage

1.  Launch the App:

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

2.  Press `` ` `` or click the `Browse operations` action to open the Operators
    list

3.  Select any of the operators listed below!

## Operators

### blur_objects

Use this operator to selectively blur objects in images based on any user-defined
query terms.
