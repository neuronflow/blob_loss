import torch


def vprint(*args):
    verbose = False
    if verbose:
        print(*args)


def compute_compound_loss(
    criterion_dict: dict,
    raw_network_outputs: torch.Tensor,
    label: torch.Tensor,
    blob_loss_mode=False,
    masked=True,
):
    """
    This computes a compound loss by looping through a criterion dict!
    """
    # vprint("outputs:", outputs)
    losses = []
    for entry in criterion_dict.values():
        name = entry["name"]
        vprint("loss name:", name)
        criterion = entry["loss"]
        weight = entry["weight"]

        sigmoid = entry["sigmoid"]
        if blob_loss_mode == False:
            vprint("computing main loss!")
            if sigmoid == True:
                sigmoid_network_outputs = torch.sigmoid(raw_network_outputs)
                individual_loss = criterion(sigmoid_network_outputs, label)
            else:
                individual_loss = criterion(raw_network_outputs, label)
        elif blob_loss_mode == True:
            vprint("computing blob loss!")
            if masked == True:  # this is the default blob loss
                if sigmoid == True:
                    sigmoid_network_outputs = torch.sigmoid(raw_network_outputs)
                    individual_loss = compute_blob_loss_multi(
                        criterion=criterion,
                        network_outputs=sigmoid_network_outputs,
                        multi_label=label,
                    )
                else:
                    individual_loss = compute_blob_loss_multi(
                        criterion=criterion,
                        network_outputs=raw_network_outputs,
                        multi_label=label,
                    )
            elif masked == False:  # without masking for ablation study
                if sigmoid == True:
                    sigmoid_network_outputs = torch.sigmoid(raw_network_outputs)
                    individual_loss = compute_no_masking_multi(
                        criterion=criterion,
                        network_outputs=sigmoid_network_outputs,
                        multi_label=label,
                    )
                else:
                    individual_loss = compute_no_masking_multi(
                        criterion=criterion,
                        network_outputs=raw_network_outputs,
                        multi_label=label,
                    )

        weighted_loss = individual_loss * weight
        losses.append(weighted_loss)

    vprint("losses:", losses)
    loss = sum(losses)
    return loss


def compute_blob_loss_multi(
    criterion,
    network_outputs: torch.Tensor,
    multi_label: torch.Tensor,
):
    """
    1. loop through elements in our batch
    2. loop through blobs per element compute loss and divide by blobs to have element loss
    2.1 we need to account for sigmoid and non/sigmoid in conjunction with BCE
    3. divide by batch length to have a correct batch loss for back prop
    """
    batch_length = multi_label.shape[0]
    vprint("batch_length:", batch_length)

    element_blob_loss = []
    # loop over elements
    for element in range(batch_length):
        if element < batch_length:
            end_index = element + 1
        elif element == batch_length:
            end_index = None

        element_label = multi_label[element:end_index, ...]
        vprint("element label shape:", element_label.shape)

        vprint("element_label:", element_label.shape)

        element_output = network_outputs[element:end_index, ...]

        # loop through labels
        unique_labels = torch.unique(element_label)
        blob_count = len(unique_labels) - 1
        vprint("found this amount of blobs in batch element:", blob_count)

        label_loss = []
        for ula in unique_labels:
            if ula == 0:
                vprint("ula is 0 we do nothing")
            else:
                # first we need one hot labels
                vprint("ula greater than 0:", ula)
                label_mask = element_label > 0
                # we flip labels
                label_mask = ~label_mask

                # we set the mask to true where our label of interest is located
                # vprint(torch.count_nonzero(label_mask))
                label_mask[element_label == ula] = 1
                # vprint(torch.count_nonzero(label_mask))
                vprint("label_mask", label_mask)
                # vprint("torch.unique(label_mask):", torch.unique(label_mask))

                the_label = element_label == ula
                the_label_int = the_label.int()
                vprint("the_label:", torch.count_nonzero(the_label))


                # debugging
                # masked_label = the_label * label_mask
                # vprint("masked_label:", torch.count_nonzero(masked_label))

                masked_output = element_output * label_mask

                try:
                    # we try with int labels first, but some losses require floats
                    blob_loss = criterion(masked_output, the_label_int)
                except:
                    # if int does not work we try float
                    blob_loss = criterion(masked_output, the_label.float())
                vprint("blob_loss:", blob_loss)

                label_loss.append(blob_loss)

        # compute mean
        vprint("label_loss:", label_loss)
        # mean_label_loss = 0
        vprint("blobs in crop:", len(label_loss))
        if not len(label_loss) == 0:
            mean_label_loss = sum(label_loss) / len(label_loss)
            # mean_label_loss = sum(label_loss) / \
            #     torch.count_nonzero(label_loss)
            vprint("mean_label_loss", mean_label_loss)
            element_blob_loss.append(mean_label_loss)

    # compute mean
    vprint("element_blob_loss:", element_blob_loss)
    mean_element_blob_loss = 0
    vprint("elements in batch:", len(element_blob_loss))
    if not len(element_blob_loss) == 0:
        mean_element_blob_loss = sum(element_blob_loss) / len(element_blob_loss)
        # element_blob_loss) / torch.count_nonzero(element_blob_loss)

    vprint("mean_element_blob_loss", mean_element_blob_loss)

    return mean_element_blob_loss


def compute_no_masking_multi(
    criterion,
    network_outputs: torch.Tensor,
    multi_label: torch.Tensor,
):
    """
    1. loop through elements in our batch
    2. loop through blobs per element compute loss and divide by blobs to have element loss
    2.1 we need to account for sigmoid and non/sigmoid in conjunction with BCE
    3. divide by batch length to have a correct batch loss for back prop
    """
    batch_length = multi_label.shape[0]
    vprint("batch_length:", batch_length)

    element_blob_loss = []
    # loop over elements
    for element in range(batch_length):
        if element < batch_length:
            end_index = element + 1
        elif element == batch_length:
            end_index = None

        element_label = multi_label[element:end_index, ...]
        vprint("element label shape:", element_label.shape)

        vprint("element_label:", element_label.shape)

        element_output = network_outputs[element:end_index, ...]

        # loop through labels
        unique_labels = torch.unique(element_label)
        blob_count = len(unique_labels) - 1
        vprint("found this amount of blobs in batch element:", blob_count)

        label_loss = []
        for ula in unique_labels:
            if ula == 0:
                vprint("ula is 0 we do nothing")
            else:
                # first we need one hot labels
                vprint("ula greater than 0:", ula)

                the_label = element_label == ula
                the_label_int = the_label.int()

                vprint("the_label:", torch.count_nonzero(the_label))

                # we compute the loss with no mask
                try:
                    # we try with int labels first, but some losses require floats
                    blob_loss = criterion(element_output, the_label_int)
                except:
                    # if int does not work we try float
                    blob_loss = criterion(element_output, the_label.float())
                vprint("blob_loss:", blob_loss)

                label_loss.append(blob_loss)

            # compute mean
            vprint("label_loss:", label_loss)
            # mean_label_loss = 0
            vprint("blobs in crop:", len(label_loss))
            if not len(label_loss) == 0:
                mean_label_loss = sum(label_loss) / len(label_loss)
                # mean_label_loss = sum(label_loss) / \
                #     torch.count_nonzero(label_loss)
                vprint("mean_label_loss", mean_label_loss)
                element_blob_loss.append(mean_label_loss)

    # compute mean
    vprint("element_blob_loss:", element_blob_loss)
    mean_element_blob_loss = 0
    vprint("elements in batch:", len(element_blob_loss))
    if not len(element_blob_loss) == 0:
        mean_element_blob_loss = sum(element_blob_loss) / len(element_blob_loss)
        # element_blob_loss) / torch.count_nonzero(element_blob_loss)

    vprint("mean_element_blob_loss", mean_element_blob_loss)

    return mean_element_blob_loss


def compute_loss(
    blob_loss_dict: dict,
    criterion_dict: dict,
    blob_criterion_dict: dict,
    raw_network_outputs: torch.Tensor,
    binary_label: torch.Tensor,
    multi_label: torch.Tensor,
):
    """
    This function computes the total loss. It has a global main loss and the blob loss term which is computed separately for each connected component. The binary_label is the binarized label for the global part. The multi label features separate integer labels for each connected component.

    Example inputs should look like:

    blob_loss_dict = {
        "main_weight": 1,
        "blob_weight": 0,
    }

    criterion_dict = {
        "bce": {
            "name": "bce",
            "loss": BCEWithLogitsLoss(reduction="mean"),
            "weight": 1.0,
            "sigmoid": False,
        },
        "dice": {
            "name": "dice",
            "loss": DiceLoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
                squared_pred=False,
            ),
            "weight": 1.0,
            "sigmoid": False,
        },
    }

    blob_criterion_dict = {
        "bce": {
            "name": "bce",
            "loss": BCEWithLogitsLoss(reduction="mean"),
            "weight": 1.0,
            "sigmoid": False,
        },
        "dice": {
            "name": "dice",
            "loss": DiceLoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
                squared_pred=False,
            ),
            "weight": 1.0,
            "sigmoid": False,
        },
    }
    """

    main_weight = blob_loss_dict["main_weight"]
    blob_weight = blob_loss_dict["blob_weight"]

    # main loss
    if main_weight > 0:
        vprint("main_weight greater than zero:", main_weight)
        # vprint("main_label:", main_label)
        main_loss = compute_compound_loss(
            criterion_dict=criterion_dict,
            raw_network_outputs=raw_network_outputs,
            label=binary_label,
            blob_loss_mode=False,
        )

    if blob_weight > 0:
        vprint("blob_weight greater than zero:", blob_weight)
        blob_loss = compute_compound_loss(
            criterion_dict=blob_criterion_dict,
            raw_network_outputs=raw_network_outputs,
            label=multi_label,
            blob_loss_mode=True,
        )

    # final loss
    if blob_weight == 0 and main_weight > 0:
        vprint(
            "main_weight:",
            main_weight,
            "// blob_weight:",
            blob_weight,
            "// computing main loss only",
        )
        loss = main_loss
        blob_loss = 0

    elif main_weight == 0 and blob_weight > 0:
        vprint(
            "main_weight:",
            main_weight,
            "// blob_weight:",
            blob_weight,
            "// computing blob loss only",
        )
        loss = blob_loss
        main_loss = 0  # we set this to 0

    elif main_weight > 0 and blob_weight > 0:
        vprint(
            "main_weight:",
            main_weight,
            "// blob_weight:",
            blob_weight,
            "// computing blob loss",
        )
        loss = main_loss * main_weight + blob_loss * blob_weight

    else:
        vprint("defaulting to equal weighted blob loss")
        loss = main_loss + blob_loss

    vprint("blob loss:", blob_loss)
    vprint("main loss:", main_loss)
    vprint("effective loss:", loss)

    return loss, main_loss, blob_loss


def get_loss_value(loss):
    if loss == 0:
        return 0

    return loss.item()
