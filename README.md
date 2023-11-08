# Media Anonymization Plugin

This plugin is a Python plugin that allows you to anonymize media in your
FiftyOne dataset by generating a copy of the images where each detection
bounding box is blurred out.


1. [Optional] Run zero-shot prediction model to generate detections field on 
    the dataset. This can be done in the app easily with the
    [Zero Shot Prediction Plugin](https://github.com/jacobmarks/zero-shot-prediction-plugin/tree/main)
2. Choose an existing `fiftyone.Detections` field (or use field from #1) to
    apply the blur effect over each bounding box to.
3. Choose a path for the new images to be saved.
4. Choose to save to an [alternate media field](https://docs.voxel51.com/user_guide/app.html#multiple-media-fields)
    or clone to a new anonymized version of the dataset.
5. Execute operator, and enjoy your new anonymized FiftyOne app consumption!

## Installation

```shell
fiftyone plugins download https://github.com/swheaton/fiftyone-media-anonymization-plugin
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

3.  Select the `blur_objects` operator listed below!

## Operators

### blur_objects

Use this operator to selectively blur objects in images based on any
`fiftyone.Detections` field in the dataset.
