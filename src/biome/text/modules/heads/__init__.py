from .classification.doc_classification import DocumentClassification
from .classification.doc_classification import DocumentClassificationConfiguration
from .classification.record_classification import RecordClassification
from .classification.record_classification import RecordClassificationConfiguration
from .classification.record_pair_classification import RecordPairClassification
from .classification.record_pair_classification import (
    RecordPairClassificationConfiguration,
)
from .classification.relation_classification import RelationClassification
from .classification.relation_classification import RelationClassificationConfiguration
from .classification.text_classification import TextClassification
from .classification.text_classification import TextClassificationConfiguration
from .language_modelling import LanguageModelling
from .language_modelling import LanguageModellingConfiguration
from .task_head import TaskHead
from .task_head import TaskHeadConfiguration
from .task_head import TaskName
from .task_head import TaskOutput
from .token_classification import TokenClassification
from .token_classification import TokenClassificationConfiguration
from .token_classification import NORMClassification

for head in [
    TextClassification,
    TokenClassification,
    NORMClassification,
    DocumentClassification,
    RecordClassification,
    LanguageModelling,
    RecordPairClassification,
    RelationClassification,
]:
    head.register(overrides=True)
