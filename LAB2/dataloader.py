import numpy as np

def read_bci_data():
    # ============================================================
    # 1. 讀取資料
    # ============================================================
    S4b_train = np.load('./lab2_EEG_classification/data/S4b_train.npz')
    X11b_train = np.load('./lab2_EEG_classification/data/X11b_train.npz')
    S4b_test = np.load('./lab2_EEG_classification/data/S4b_test.npz')
    X11b_test = np.load('./lab2_EEG_classification/data/X11b_test.npz')

    # ============================================================
    # 2. 合併 train/test
    # ============================================================
    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)



    # ============================================================
    # 3. Label 調整成 0 / 1
    # ============================================================
    train_label = train_label - 1
    test_label = test_label - 1

    # ============================================================
    # 4. 維度轉換 (確保 shape = (N, 1, 2, 750))
    # ============================================================
    print("Raw train_data shape:", train_data.shape)

    # 原始 shape 為 (N, 750, 2)，轉換為 (N, 1, 2, 750)
    train_data = np.transpose(train_data, (0, 2, 1))
    test_data = np.transpose(test_data, (0, 2, 1))
    train_data = (train_data - np.mean(train_data, axis=-1, keepdims=True)) \
             / (np.std(train_data, axis=-1, keepdims=True) + 1e-6)
    test_data  = (test_data - np.mean(test_data, axis=-1, keepdims=True)) \
                / (np.std(test_data, axis=-1, keepdims=True) + 1e-6)
    train_data = np.expand_dims(train_data, axis=1)
    test_data = np.expand_dims(test_data, axis=1)
    
    # ============================================================
    # 5. 處理 NaN
    # ============================================================
    if np.isnan(train_data).any():
        print("⚠️ NaN detected in train data, replacing with mean")
        train_data[np.isnan(train_data)] = np.nanmean(train_data)
    if np.isnan(test_data).any():
        print("⚠️ NaN detected in test data, replacing with mean")
        test_data[np.isnan(test_data)] = np.nanmean(test_data)

    # ============================================================
    # 6. 結果確認
    # ============================================================
    print("✅ Final shapes:")
    print("train_data:", train_data.shape)
    print("train_label:", train_label.shape)
    print("test_data:", test_data.shape)
    print("test_label:", test_label.shape)

    return train_data, train_label, test_data, test_label
