"""
I/O operators.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import cv2
import numpy as np
from PIL import Image

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou


def _set_progress(ctx, progress, label=None, delegated=False):
    if delegated:
        return None
    return ctx.trigger("set_progress", dict(progress=progress, label=label))


def process_image_results(
    image, image_results, query_string, output_dir, store_detections
):
    file_path = image.filename
    detections = []
    if image_results["boxes"].any():
        # Use NP array in BGR, which cv2 likes
        cv2_image = np.asarray(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        for box, score, label_num in zip(
            image_results["boxes"],
            image_results["scores"],
            image_results["labels"],
        ):
            x0, y0, x1, y1 = box.tolist()
            slice_y, slice_x = slice(
                int(y0 * image.height), int(y1 * image.height)
            ), slice(int(x0 * image.width), int(x1 * image.width))
            roi = cv2_image[slice_y, slice_x]

            if roi.any():
                # Gaussian blur approach
                # roi = cv2.GaussianBlur(roi, (51, 51), 0)

                # Perform stack blur. Faster than gaussian, looks better
                #   than pixelation.
                #   stackBlur() can segfault if ksize is too big relative
                #   to the roi so cap it to a smaller amount.
                ksize = min(101, min(roi.shape[0:2]) * 2 + 1)
                roi = cv2.stackBlur(roi, ksize=(ksize, ksize))

                # Pixelation approach
                # height, width, _ = roi.shape
                # roi = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                # roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_NEAREST)

                # impose this blurred image on original image to get final image
                cv2_image[slice_y, slice_x] = roi

                # Create FO detection for this box
                if store_detections:
                    box_width = x1 - x0
                    box_height = y1 - y0
                    label = query_string
                    if isinstance(query_string, list):
                        label = query_string[label_num]
                    detections.append(
                        fo.Detection(
                            label=label,
                            confidence=score,
                            bounding_box=[
                                x0,
                                y0,
                                box_width,
                                box_height,
                            ],
                        )
                    )

        # Convert back to PIL.Image for saving
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_image)

    base_name = os.path.basename(file_path)
    root, ext = os.path.splitext(base_name)
    anon_file_path = os.path.join(output_dir, f"{root}_anon{ext}")
    image.save(anon_file_path)

    return detections, anon_file_path


def process_batch_results(
    results, batch, query_string, output_dir, store_detections
):
    out_files = []
    batch_detections = []
    for i, image_results in enumerate(results):
        detections, anon_file_path = process_image_results(
            batch[i], image_results, query_string, output_dir, store_detections
        )
        if store_detections:
            batch_detections.append(fo.Detections(detections=detections))
        out_files.append(anon_file_path)

    return out_files, batch_detections


def anonymize_batch(
    processor, model, batch, query_string, output_dir, store_detections
):
    inputs = processor(
        text=[query_string] * len(batch), images=batch, return_tensors="pt"
    )
    # inputs.to(device)

    outputs = model(**inputs)
    # outputs.to(device)

    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=0.05
    )

    return process_batch_results(
        results, batch, query_string, output_dir, store_detections
    )


def anonymize_media(
    ctx,
    target_view,
    query_string,
    output_dir,
    store_detections,
    delegated,
    media_path,
):
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    # Comma separated queries
    if "," in query_string:
        query_string = [s.strip() for s in query_string.split(",")]

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32"
    )
    # device = torch.device("mps")
    # processor.to(device)
    # model.to(device)

    file_paths = target_view.values("filepath")
    images = [Image.open(f) for f in file_paths]

    batch_size = 1
    yield _set_progress(ctx, 0, f"Anonymized 0 of {len(images)}", delegated)

    all_detections = []
    new_files = []
    for batch_start in range(0, len(images), batch_size):
        batch = images[batch_start : batch_start + batch_size]
        batch_files, batch_detections = anonymize_batch(
            processor, model, batch, query_string, output_dir, store_detections
        )
        new_files += batch_files
        all_detections += batch_detections

        progress = (batch_start + batch_size) / len(images)
        label = f"Anonymized {batch_start + batch_size} of {len(images)}"
        yield _set_progress(ctx, progress, label, delegated)

    if store_detections:
        target_view.set_values("anonymized_detections", all_detections)

    target_view.set_values(media_path, new_files)


def get_target_view(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Process the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations "
                    "for more information"
                )
            ),
        )


class AnonymizeMedia(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="blur_objects",
            label="Blur Objects in Images",
            dynamic=True,
            execute_as_generator=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _anonymize_media_inputs(ctx, inputs)
        if ready:
            _execution_mode(ctx, inputs)

        return types.Property(
            inputs, view=types.View(label="Anonymize Images")
        )

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        # Params
        target = ctx.params.get("target", None)
        query_string = ctx.params["query_string"]
        media_path = ctx.params["media_path"]
        output_dir = ctx.params["output_dir"]["absolute_path"]
        overwrite = ctx.params.get("overwrite", False)
        store_detections = ctx.params.get("store_detections", True)
        delegated = ctx.params.get("delegated", False)
        target_view = _get_target_view(ctx, target)

        if not overwrite:
            target_view = target_view.exists(media_path, False)

        num_total = len(target_view)
        with fou.ProgressBar(total=num_total) as pb:
            for update in pb(
                anonymize_media(
                    ctx,
                    target_view,
                    query_string,
                    output_dir,
                    store_detections,
                    delegated,
                    media_path,
                )
            ):
                yield update

        # Update media fields
        if media_path not in ctx.dataset.app_config.media_fields:
            ctx.dataset.app_config.media_fields.append(media_path)
        ctx.dataset.app_config.grid_media_field = media_path
        ctx.dataset.app_config.modal_media_field = media_path

        # Save and reload
        ctx.dataset.save()
        yield ctx.trigger("reload_dataset")


def _parse_path(ctx, key):
    value = ctx.params.get(key, None)
    return value.get("absolute_path", None) if value else None


def _anonymize_media_inputs(ctx, inputs):
    # Dataset, view, or selection?
    target_view = get_target_view(ctx, inputs)

    target = ctx.params.get("target", None)
    if target == "SELECTED_SAMPLES":
        target_str = "selection"
    elif target == "CURRENT_VIEW":
        target_str = "current view"
    else:
        target_str = "dataset"

    # Query string
    query_string_selector = types.AutocompleteView()
    for preset in ["human face", "person", "license plate"]:
        query_string_selector.add_choice(preset, label=preset.title())
    inputs.str(
        "query_string",
        required=True,
        label="Query String",
        description="Text query string to search for, can be a word or phrase. Pass a comma ',' separated list to search for multiple classes; e.g., horse,cat,dog",
        view=query_string_selector,
    )

    query_string = ctx.params.get("query_string", None)
    if not query_string:
        return False

    field_selector = types.AutocompleteView()
    field_selector.add_choice(
        "anonymized_filepath", label="anonymized_filepath"
    )
    for field in target_view.get_field_schema(ftype=fo.StringField):
        if field == "filepath":
            continue

        field_selector.add_choice(field, label=field)

    inputs.str(
        "media_path",
        required=True,
        label="Anonymized media field",
        description=(
            "Provide the name of a new or existing field in which to "
            "store the anonymized media paths"
        ),
        view=field_selector,
    )

    media_path = ctx.params.get("media_path", None)

    if media_path is None:
        return False

    file_explorer = types.FileExplorerView(
        choose_dir=True,
        button_label="Choose a directory...",
    )
    inputs.file(
        "output_dir",
        required=True,
        label="Output directory",
        description=(
            "Choose a new or existing directory into which to write the "
            "generated anonymized media"
        ),
        view=file_explorer,
    )
    output_dir = _parse_path(ctx, "output_dir")

    if output_dir is None:
        return False

    inputs.bool(
        "store_detections",
        default=True,
        label="Store detections used for anonymization?",
        description="Whether to store output detections from this model run as a field on the dataset",
        view=types.SwitchView(),
    )

    inputs.bool(
        "overwrite",
        default=False,
        label="Regenerate anonymized media?",
        description=f"Whether to regenerate anonymized media for samples that already have their '{media_path} field populated'",
        view=types.SwitchView(),
    )

    overwrite = ctx.params.get("overwrite", False)

    if overwrite:
        n = len(target_view)
        if n > 0:
            label = f"Found {n} samples to (re)generate anonymized media for"
        else:
            label = f"Your {target_str} is empty"
    else:
        n = len(target_view.exists(media_path, False))
        if n > 0:
            label = f"Found {n} samples that need anonymized media generated"
        else:
            label = (
                f"All samples in your {target_str} already have "
                "anonymized media generated"
            )

    if n > 0:
        inputs.view("status1", types.Notice(label=label))
    else:
        status1 = inputs.view("status1", types.Warning(label=label))
        status1.invalid = True
        return False

    return True


def register(p):
    p.register(AnonymizeMedia)
