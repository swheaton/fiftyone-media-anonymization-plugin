"""
I/O operators.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import functools
import os

import cv2
from PIL import Image

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou


def stack_blur(roi):
    """Perform stack blur. Faster than gaussian, looks better than pixelation"""
    #   stackBlur() can segfault if ksize is too big relative
    #   to the roi so cap it to a smaller amount.
    ksize = min(101, min(roi.shape[0:2]) * 2 + 1)
    roi = cv2.stackBlur(roi, ksize=(ksize, ksize))
    return roi


def pixelate(roi):
    """Pixelation approach"""
    height, width, _ = roi.shape
    roi = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
    roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_NEAREST)
    return roi


def gaussian_blur(roi):
    """Use gaussian blur to blur"""
    # Gaussian blur approach
    roi = cv2.GaussianBlur(roi, (51, 51), 0)
    return roi


def anonymize_media(
    target_view,
    output_dir,
    media_path,
    detections_field,
    blur_method,
    progress_callback,
):
    anon_file_paths = {}
    filepaths, boxes_arr = target_view.values(
        ("filepath", f"{detections_field}.detections.bounding_box")
    )
    total = len(target_view)
    for (i, (filepath, boxes)) in enumerate(zip(filepaths, boxes_arr)):

        cv2_image = cv2.imread(filepath)

        for box in boxes:
            x0, y0, width, height = box
            total_height, total_width, _ = cv2_image.shape

            slice_y = slice(int(y0 * total_height), int((y0 + height) * total_height))
            slice_x = slice(int(x0 * total_width), int((x0 + width) * total_width))

            roi = cv2_image[slice_y, slice_x]

            if roi.any():
                if blur_method == "gaussian":
                    roi = gaussian_blur(roi)
                elif blur_method == "stack":
                    roi = stack_blur(roi)
                elif blur_method == "pixelate":
                    roi = pixelate(roi)
                else:
                    raise ValueError(f"Unknown blur method '{blur_method}'")

                # impose this blurred image on original image to get final image
                cv2_image[slice_y, slice_x] = roi

        # Convert back to PIL.Image for saving
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_image)

        base_name = os.path.basename(filepath)
        root, ext = os.path.splitext(base_name)
        anon_file_path = os.path.join(output_dir, f"{root}_anon{ext}")
        image.save(anon_file_path)
        anon_file_paths[filepath] = anon_file_path

        yield progress_callback(
            progress=i / total, label=f"{i}/{total} samples anonymized"
        )

    target_view.set_values(media_path, anon_file_paths, key_field="filepath")
    yield progress_callback(progress=1, label="All samples anonymized")


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
            label="Anonymize/Blur Objects in Images",
            dynamic=True,
            execute_as_generator=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        ready = _anonymize_media_inputs(ctx, inputs)
        if ready:
            _execution_mode(ctx, inputs)

        return types.Property(inputs, view=types.View(label="Anonymize Images"))

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    async def execute(self, ctx):
        # Params
        target = ctx.params.get("target", None)
        detections_field = ctx.params["detections_field"]
        mode = ctx.params.get("mode")
        output_dir = ctx.params["output_dir"]["absolute_path"]
        overwrite = ctx.params.get("overwrite", False)
        delegated = ctx.params.get("delegated", False)
        target_view = _get_target_view(ctx, target)

        update_ctx_dataset = False
        if mode == "alt":
            media_path = ctx.params["media_path"]
            update_ctx_dataset = True
            if not overwrite:
                target_view = target_view.exists(media_path, False)
        elif mode == "clone":
            clone_dataset = ctx.params["clone_dataset"]
            target_view = target_view.clone(clone_dataset)
            media_path = "filepath"
        else:
            raise ValueError("Invalid mode")

        def _set_progress(ctx, progress, label=None, delegated=False):
            if delegated:
                return None
            return ctx.trigger("set_progress", dict(progress=progress, label=label))

        num_total = len(target_view)
        progress_callback = functools.partial(
            _set_progress, ctx=ctx, delegated=delegated
        )
        with fou.ProgressBar(total=num_total) as pb:
            for update in pb(
                anonymize_media(
                    target_view,
                    output_dir,
                    media_path,
                    detections_field,
                    "stack",
                    progress_callback,
                )
            ):
                yield update

        # Update media fields
        if update_ctx_dataset:
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
    if ctx.dataset.media_type != "image":
        label = "Only image datasets currently supported."
        status1 = inputs.view("status1", types.Warning(label=label))
        status1.invalid = True
        return False

    # Dataset, view, or selection?
    target_view = get_target_view(ctx, inputs)

    target = ctx.params.get("target", None)
    if target == "SELECTED_SAMPLES":
        target_str = "selection"
    elif target == "CURRENT_VIEW":
        target_str = "current view"
    else:
        target_str = "dataset"

    detection_field_selector = types.AutocompleteView()
    for field in target_view.get_field_schema(embedded_doc_type=fo.Detections):
        detection_field_selector.add_choice(field, label=field)

    inputs.str(
        "detections_field",
        required=True,
        label="Detections field",
        description=(
            "Choose an existing Detections field to use for anonymization "
            "bounding boxes."
        ),
        view=detection_field_selector,
    )

    detections_field = ctx.params.get("detections_field", None)

    if detections_field is None:
        return False

    mode_selector = types.Choices()
    mode_selector.add_choice("alt", label="Create alternative media source")
    mode_selector.add_choice("clone", label="Clone into new dataset")
    inputs.str(
        "mode",
        required=True,
        label="",
        description="",
        default="alt",
        view=mode_selector,
    )

    mode = ctx.params.get("mode")
    if not mode:
        return False
    if mode == "alt":
        field_selector = types.AutocompleteView()
        field_selector.add_choice("anonymized_filepath", label="anonymized_filepath")
        for field in target_view.get_field_schema(ftype=fo.StringField):
            if field in {"anonymized_filepath", "filepath"}:
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

        inputs.bool(
            "overwrite",
            default=False,
            label="Regenerate anonymized media?",
            description=(
                "Whether to regenerate anonymized media for samples that "
                f"already have their '{media_path}' field populated"
            ),
            view=types.SwitchView(),
        )

    elif mode == "clone":
        inputs.str("clone_dataset", required=True, label="Clone dataset name")

        if "clone_dataset" not in ctx.params:
            return False
    else:
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

    if not output_dir:
        return False

    overwrite = ctx.params.get("overwrite", False)

    if overwrite or mode == "clone":
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
