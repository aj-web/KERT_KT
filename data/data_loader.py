"""
Data loading and preprocessing for Knowledge Tracing datasets
Supports ASSIST09, ASSIST17, and Junyi datasets
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class KTDataLoader:
    """Knowledge Tracing Data Loader"""

    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_assist09(self):
        """Download ASSIST09 dataset"""
        print("Downloading ASSIST09 dataset...")
        url = "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/2009-2010-assistments-data.zip?attredirects=0&d=1"
        zip_path = os.path.join(self.data_dir, 'assist09.zip')

        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("ASSIST09 dataset downloaded successfully")
        except:
            print("ASSIST09 download failed, using synthetic data for demo")
            self._create_synthetic_assist09()

    def download_assist17(self):
        """Download ASSIST17 dataset"""
        print("Downloading ASSIST17 dataset...")
        url = "https://sites.google.com/site/assistmentsdata/home/2017-assistments-data/2017-Assistments-data.zip?attredirects=0&d=1"
        zip_path = os.path.join(self.data_dir, 'assist17.zip')

        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("ASSIST17 dataset downloaded successfully")
        except:
            print("ASSIST17 download failed, using synthetic data for demo")
            self._create_synthetic_assist17()

    def download_junyi(self):
        """Download Junyi dataset"""
        print("Downloading Junyi dataset...")
        # Junyi dataset might require manual download from their website
        print("Junyi dataset requires manual download from Junyi Academy website")
        print("Using synthetic data for demo")
        self._create_synthetic_junyi()

    def _create_synthetic_assist09(self):
        """Create synthetic ASSIST09-like data with realistic learning patterns"""
        np.random.seed(42)

        # Simulate ASSIST09 characteristics
        n_students = 4151
        n_questions = 17751
        n_concepts = 124
        total_interactions = 325637

        # 步骤1：创建Q-matrix（question到concept的固定映射）
        print("Creating Q-matrix...")
        q_matrix = {}  # question_id -> concept_id
        for q_id in range(n_questions):
            # 每个question对应1个concept（简化）
            concept_id = q_id % n_concepts
            q_matrix[q_id] = concept_id

        # 步骤2：为每个question分配难度
        question_difficulty = np.random.normal(0, 1, n_questions)  # 标准正态分布

        # 步骤3：生成学生交互数据
        data = []
        student_id = 0
        interaction_count = 0

        while interaction_count < total_interactions:
            # 初始化学生对每个concept的掌握水平
            student_ability = np.random.normal(0, 1, n_concepts)  # 每个concept的掌握水平

            # Each student has varying sequence length
            seq_length = np.random.poisson(78) + 5  # Mean around 78

            for i in range(seq_length):
                if interaction_count >= total_interactions:
                    break

                # 随机选择一个question
                question_id = np.random.randint(0, n_questions)
                concept_id = q_matrix[question_id]  # 固定映射

                # 基于IRT模型计算正确概率：P(correct) = sigmoid(ability - difficulty)
                ability = student_ability[concept_id]
                difficulty = question_difficulty[question_id]
                prob_correct = 1 / (1 + np.exp(-(ability - difficulty)))

                # 生成答案
                correct = np.random.binomial(1, prob_correct)

                data.append({
                    'student_id': student_id,
                    'question_id': question_id,
                    'concept_id': concept_id,
                    'correct': correct,
                    'timestamp': interaction_count
                })

                # 学习效应：答对后能力提升
                if correct == 1:
                    student_ability[concept_id] += 0.05  # 小幅提升
                else:
                    student_ability[concept_id] += 0.01  # 答错也有小幅提升

                interaction_count += 1

            student_id += 1
            if student_id >= n_students:
                break

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data_dir, 'assist09.csv'), index=False)
        print(f"Synthetic ASSIST09 data created with Q-matrix (avg correct rate: {df['correct'].mean():.3f})")

    def _create_synthetic_assist17(self):
        """Create synthetic ASSIST17-like data with realistic learning patterns"""
        np.random.seed(43)

        n_students = 1709
        n_questions = 3162
        n_concepts = 102
        total_interactions = 942816

        # 创建Q-matrix
        print("Creating Q-matrix for ASSIST17...")
        q_matrix = {q_id: q_id % n_concepts for q_id in range(n_questions)}
        question_difficulty = np.random.normal(0, 1, n_questions)

        data = []
        student_id = 0
        interaction_count = 0

        while interaction_count < total_interactions:
            student_ability = np.random.normal(0, 1, n_concepts)
            seq_length = np.random.poisson(551) + 5  # Mean around 551

            for i in range(seq_length):
                if interaction_count >= total_interactions:
                    break

                question_id = np.random.randint(0, n_questions)
                concept_id = q_matrix[question_id]

                ability = student_ability[concept_id]
                difficulty = question_difficulty[question_id]
                prob_correct = 1 / (1 + np.exp(-(ability - difficulty)))
                correct = np.random.binomial(1, prob_correct)

                data.append({
                    'student_id': student_id,
                    'question_id': question_id,
                    'concept_id': concept_id,
                    'correct': correct,
                    'timestamp': interaction_count
                })

                if correct == 1:
                    student_ability[concept_id] += 0.05
                else:
                    student_ability[concept_id] += 0.01

                interaction_count += 1

            student_id += 1

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data_dir, 'assist17.csv'), index=False)
        print(f"Synthetic ASSIST17 data created with Q-matrix (avg correct rate: {df['correct'].mean():.3f})")

    def _create_synthetic_junyi(self):
        """Create synthetic Junyi-like data with realistic learning patterns"""
        np.random.seed(44)

        n_students = 10000
        n_questions = 25925
        n_concepts = 835
        total_interactions = 1062631

        # 创建Q-matrix
        print("Creating Q-matrix for Junyi...")
        q_matrix = {q_id: q_id % n_concepts for q_id in range(n_questions)}
        question_difficulty = np.random.normal(0, 1, n_questions)

        data = []
        student_id = 0
        interaction_count = 0

        while interaction_count < total_interactions:
            student_ability = np.random.normal(0, 1, n_concepts)
            seq_length = np.random.poisson(1062) + 5  # Mean around 1062

            for i in range(seq_length):
                if interaction_count >= total_interactions:
                    break

                question_id = np.random.randint(0, n_questions)
                concept_id = q_matrix[question_id]

                ability = student_ability[concept_id]
                difficulty = question_difficulty[question_id]
                prob_correct = 1 / (1 + np.exp(-(ability - difficulty)))
                correct = np.random.binomial(1, prob_correct)

                data.append({
                    'student_id': student_id,
                    'question_id': question_id,
                    'concept_id': concept_id,
                    'correct': correct,
                    'timestamp': interaction_count
                })

                if correct == 1:
                    student_ability[concept_id] += 0.05
                else:
                    student_ability[concept_id] += 0.01

                interaction_count += 1

            student_id += 1

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data_dir, 'junyi.csv'), index=False)
        print(f"Synthetic Junyi data created with Q-matrix (avg correct rate: {df['correct'].mean():.3f})")

    def preprocess_data(self, dataset_name, min_interactions=5, force_download=False):
        """Preprocess the dataset according to paper specifications

        Args:
            dataset_name: name of the dataset ('assist09', 'assist17', 'junyi')
            min_interactions: minimum interactions required per student
            force_download: if True, force re-download even if file exists
        """

        file_path = os.path.join(self.data_dir, f'{dataset_name}.csv')

        # Check if dataset file exists
        if not os.path.exists(file_path) or force_download:
            if force_download:
                print(f"Force downloading {dataset_name} dataset...")
            else:
                print(f"Dataset {dataset_name}.csv not found, downloading...")

            # Try to download or create synthetic data
            if dataset_name == 'assist09':
                self.download_assist09()
            elif dataset_name == 'assist17':
                self.download_assist17()
            elif dataset_name == 'junyi':
                self.download_junyi()
        else:
            print(f"Dataset {dataset_name}.csv already exists, skipping download.")

        # Load data
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} not found after download attempt.")

        df = pd.read_csv(file_path)

        # Data cleaning (论文4.1节)
        # 1. Remove students with minimum interactions
        student_counts = df.groupby('student_id').size()
        valid_students = student_counts[student_counts >= min_interactions].index
        df = df[df['student_id'].isin(valid_students)]
        
        # 2. Remove abnormal response times (论文4.1节：删除答题时间<1秒或>1小时的记录)
        # Check for response time columns (答题耗时，不是绝对时间戳)
        time_columns = ['response_time', 'time_taken', 'time_spent', 'duration', 'elapsed_time']
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is not None:
            # Filter out records with response time < 1 second or > 3600 seconds (1 hour)
            df = df[(df[time_col] >= 1) & (df[time_col] <= 3600)]
        # Note: If only 'timestamp' exists (absolute time), we cannot filter by response duration
        # This would require computing time differences between consecutive records
        
        # 3. Sort by timestamp (论文4.1节：时序排序)
        df = df.sort_values(['student_id', 'timestamp']).reset_index(drop=True)

        # Create concept-question mapping (Q-matrix)
        n_questions = df['question_id'].max() + 1
        n_concepts = df['concept_id'].max() + 1

        # Build Q-matrix: for simplicity, each question maps to one concept
        # In real datasets, this might be many-to-many
        q_matrix = np.zeros((n_questions, n_concepts))
        for _, row in df.iterrows():
            q_matrix[int(row['question_id']), int(row['concept_id'])] = 1

        # Build knowledge graph adjacency matrix
        concept_adjacency = self._build_concept_graph(df, n_concepts)

        # Split data: 严格时序划分 (论文4.2.2节)
        # 对每个学生，前70%→训练，中间10%→验证，最后20%→测试
        train_list = []
        val_list = []
        test_list = []
        
        for student_id, group in df.groupby('student_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            n_interactions = len(group)
            
            # 计算划分点
            train_end = int(n_interactions * 0.7)
            val_end = int(n_interactions * 0.8)  # 70% + 10% = 80%
            
            # 严格时序划分
            train_list.append(group.iloc[:train_end])
            val_list.append(group.iloc[train_end:val_end])
            test_list.append(group.iloc[val_end:])
        
        train_data = pd.concat(train_list, ignore_index=True)
        val_data = pd.concat(val_list, ignore_index=True)
        test_data = pd.concat(test_list, ignore_index=True)

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'q_matrix': q_matrix,
            'concept_graph': concept_adjacency,
            'n_questions': n_questions,
            'n_concepts': n_concepts,
            'n_students': len(valid_students)
        }

    def _build_concept_graph(self, df, n_concepts):
        """Build concept relationship graph based on co-occurrence"""
        adjacency = np.zeros((n_concepts, n_concepts))

        # Count concept co-occurrences in questions
        concept_pairs = {}
        for _, group in df.groupby('question_id'):
            concepts_in_question = group['concept_id'].unique()
            for i in range(len(concepts_in_question)):
                for j in range(i+1, len(concepts_in_question)):
                    pair = tuple(sorted([concepts_in_question[i], concepts_in_question[j]]))
                    concept_pairs[pair] = concept_pairs.get(pair, 0) + 1

        # Build adjacency matrix
        for (c1, c2), weight in concept_pairs.items():
            adjacency[c1, c2] = weight
            adjacency[c2, c1] = weight

        # Normalize adjacency matrix
        degrees = np.sum(adjacency, axis=1)
        degrees[degrees == 0] = 1  # Avoid division by zero
        adjacency = adjacency / degrees[:, np.newaxis]

        return adjacency


def prepare_all_datasets():
    """Prepare all three datasets"""
    loader = KTDataLoader()

    datasets = {}
    for dataset_name in ['assist09', 'assist17', 'junyi']:
        print(f"\nProcessing {dataset_name}...")
        datasets[dataset_name] = loader.preprocess_data(dataset_name)

        print(f"{dataset_name} statistics:")
        print(f"  Students: {datasets[dataset_name]['n_students']}")
        print(f"  Questions: {datasets[dataset_name]['n_questions']}")
        print(f"  Concepts: {datasets[dataset_name]['n_concepts']}")
        print(f"  Train interactions: {len(datasets[dataset_name]['train'])}")
        print(f"  Val interactions: {len(datasets[dataset_name]['val'])}")
        print(f"  Test interactions: {len(datasets[dataset_name]['test'])}")

    return datasets


if __name__ == "__main__":
    datasets = prepare_all_datasets()

    # Save processed data
    import pickle
    with open('./data/processed_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)

    print("\nAll datasets processed and saved!")
