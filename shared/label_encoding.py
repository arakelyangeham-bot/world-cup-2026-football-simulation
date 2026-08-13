# label_encoding.py

from sklearn.preprocessing import LabelEncoder


def encode_labels(y, y_train, y_test):
    """
    Encode string class labels into integer labels.

    Returns:
        y_encoded
        y_train_encoded
        y_test_encoded
        encoded_labels
        label_mapping
        label_encoder
    """

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    encoded_labels = list(range(len(label_encoder.classes_)))

    label_mapping = {
        int(i): label
        for i, label in enumerate(label_encoder.classes_)
    }

    return (
        y_encoded,
        y_train_encoded,
        y_test_encoded,
        encoded_labels,
        label_mapping,
        label_encoder,
    )