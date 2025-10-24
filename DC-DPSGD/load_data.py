import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import re


def load_tabular_local(name, path):
    """
    name: 'adult' | 'bank' | 'credit'
    path: 本地 CSV 文件路径
    """
    name = name.lower()
    
    if name == 'adult':
        ADULT_COLS = [
            'age','workclass','fnlwgt','education','education-num','marital-status',
            'occupation','relationship','race','sex','capital-gain','capital-loss',
            'hours-per-week','native-country','label'
        ]
        train_path = os.path.join(path, "adult.data")
        test_path  = os.path.join(path, "adult.test")

        # 读训练集（无表头）
        df_tr = pd.read_csv(train_path, header=None, names=ADULT_COLS,
                            na_values="?", skipinitialspace=True)

        # 读测试集（有些版本首行是注释；标签常带句号）
        df_te = pd.read_csv(test_path, header=None, names=ADULT_COLS,
                            na_values="?", skipinitialspace=True, comment='|')

        # 去空白
        for df_tmp in (df_tr, df_te):
            for c in df_tmp.columns:
                if df_tmp[c].dtype == object:
                    df_tmp[c] = df_tmp[c].astype(str).str.strip()


        # 统一标签：'>50K' / '>50K.' → 1，其余 0
        def binarize(s: pd.Series):
            s = s.astype(str).str.replace(".", "", regex=False).str.strip()
            return (s == ">50K").astype(int)
        df_tr["label"] = binarize(df_tr["label"])
        df_te["label"] = binarize(df_te["label"])

        # 丢缺失（或改成填充）
        df_tr = df_tr.dropna().reset_index(drop=True)
        df_te = df_te.dropna().reset_index(drop=True)

        # 需要的话，这里直接返回 DataFrame；或做预处理：
        X_train_raw = df_tr.drop(columns=["label"])
        y_train = df_tr["label"].to_numpy(int)
        X_test_raw  = df_te.drop(columns=["label"])
        y_test  = df_te["label"].to_numpy(int)

        # 数值/类别分列 + 标准化/OneHot
        num_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_train_raw.columns if c not in num_cols]
        pre = ColumnTransformer([
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])
        X_train = pre.fit_transform(X_train_raw)
        X_test  = pre.transform(X_test_raw)
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        return X_train, X_test, y_train, y_test
    
    elif name == 'bank':
        df = pd.read_csv(path, sep=';')
        df.columns = [c.strip() for c in df.columns]
        df["label"] = (df["y"].astype(str).str.strip() == "yes").astype(int)
        df = df.drop(columns=["y"])
    
    elif name == 'credit':
        df = pd.read_excel(path)
        if "default.payment.next.month" in df.columns:
            df = df.rename(columns={"default.payment.next.month": "label"})
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

    elif name == 'yeast':
        X, y = fetch_openml("yeast", version=1, return_X_y=True, as_frame=False)
        print(f"Shape: {X.shape}, Labels: {np.unique(y)}")

        # 标签编码为 0~9
        le = LabelEncoder()
        y = le.fit_transform(y)

        # 归一化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    elif name == 'product':
        TEXT_COL = "Product Title"   # 文本列
        LABEL_COL = "Category Label" # 预测标签列，可改为 "Cluster Label"
        df = pd.read_csv(path)
        print("原始列名：", df.columns.tolist())
        print(f"数据样本数: {len(df)}")

        # 自动去除列名首尾空格
        df.columns = [c.strip() for c in df.columns]

        # 去除重复行和缺失
        df = df.drop_duplicates(subset=[TEXT_COL])
        df = df.dropna(subset=[TEXT_COL, LABEL_COL])
        df = shuffle(df, random_state=42).reset_index(drop=True)

        print(f"清洗后样本数: {len(df)}")
        print(df.head(3))

        def clean_text(s: str) -> str:
            s = str(s).lower()
            s = s.replace("_", " ")
            s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
            s = " ".join(s.split())  # 压缩多空格
            return s

        df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

        label_encoder = LabelEncoder()
        df[LABEL_COL] = label_encoder.fit_transform(df[LABEL_COL])
        print(f"类别数: {len(label_encoder.classes_)}")
        print("前5个类别映射：", dict(zip(label_encoder.classes_[:5], range(5))))

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),     # uni-gram + bi-gram
            min_df=3,               # 至少出现3次
            max_features=30000,     # 限制最大特征数
            sublinear_tf=True
        )

        
    else:
        raise ValueError("Unsupported dataset name.")

    # 清理缺失值
    df = df.dropna()
    print(df.columns.tolist())

    if name == 'bank':
        y = df["label"].astype(int).values
        X = df.drop(columns=["label"])

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        pre = ColumnTransformer([
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train = pre.fit_transform(X_train)
        X_test = pre.transform(X_test)
    elif name == 'product':
        X = vectorizer.fit_transform(df[TEXT_COL])
        y = df[LABEL_COL].values

        print(f"TF-IDF 特征维度: {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    else:
        df = df[df['Y'] != 'default payment next month']
        print(df['Y'].unique())
        y = df["Y"].astype(int).values
        X = df.drop(columns=["Y"])

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        pre = ColumnTransformer([
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])
        
        X_train = pre.fit_transform(X_train).toarray()
        X_test = pre.transform(X_test).toarray()
        

    

    return X_train, X_test, y_train, y_test