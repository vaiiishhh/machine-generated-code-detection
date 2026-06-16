# -*- coding: utf-8 -*-
"""catboost_authorship.py
Authorship identification variant of the CatBoost pipeline.
Task: given a code snippet, predict WHICH model generated it
      (or whether it is human-written).

Labels  (MODEL_LABEL_MAP):
  0 → human
  1 → llama3.1   (llama 3.1 / meta-llama / llama-3.1-*)
  2 → qwen1.5    (qwen1.5 / qwen-*)
  3 → codellama  (codellama / code-llama)
  4 → nxcode     (nxcode / nx-code)
  5 → gpt        (gpt-* / openai)
"""

!pip install catboost -q
!pip install javalang
!pip install tree-sitter==0.21.3 tree-sitter-languages==1.10.2

import ast
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict
import subprocess, shutil, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False

TREESITTER_AVAILABLE = False
CPP_LANGUAGE = None
CPP_TS_PARSER = None

def _try_init_treesitter():
    global TREESITTER_AVAILABLE, CPP_LANGUAGE, CPP_TS_PARSER
    try:
        import tree_sitter_languages as _tsl
        from tree_sitter import Language as _Lang, Parser as _Parser
        _ptr = _tsl.get_language("cpp")
        try:
            CPP_LANGUAGE = _Lang(_ptr, "cpp")
        except TypeError:
            CPP_LANGUAGE = _ptr
        CPP_TS_PARSER = _Parser()
        CPP_TS_PARSER.set_language(CPP_LANGUAGE)
        TREESITTER_AVAILABLE = True
        return
    except Exception:
        pass
    try:
        from tree_sitter_languages import get_language as _gl, get_parser as _gp
        CPP_LANGUAGE  = _gl("cpp")
        CPP_TS_PARSER = _gp("cpp")
        TREESITTER_AVAILABLE = True
        return
    except Exception:
        pass
    try:
        import tree_sitter_cpp as _tscpp
        from tree_sitter import Language as _Lang, Parser as _Parser
        CPP_LANGUAGE = _Lang(_tscpp.language())
        CPP_TS_PARSER = _Parser(CPP_LANGUAGE)
        TREESITTER_AVAILABLE = True
        return
    except Exception:
        pass
    try:
        import tree_sitter_cpp as _tscpp
        from tree_sitter import Language as _Lang, Parser as _Parser
        CPP_LANGUAGE = _Lang(_tscpp.language(), "cpp")
        CPP_TS_PARSER = _Parser()
        CPP_TS_PARSER.set_language(CPP_LANGUAGE)
        TREESITTER_AVAILABLE = True
        return
    except Exception:
        pass

_try_init_treesitter()

import ast as _pyast
PYTHON_NODES = sorted([
    name for name in dir(_pyast)
    if isinstance(getattr(_pyast, name), type)
    and issubclass(getattr(_pyast, name), _pyast.AST)
    and not name.startswith("_")
])

JAVA_NODES = [
    "CompilationUnit","Import","Documented","Declaration","TypeDeclaration",
    "PackageDeclaration","ClassDeclaration","EnumDeclaration","InterfaceDeclaration",
    "AnnotationDeclaration","Type","BasicType","ReferenceType","TypeArgument",
    "TypeParameter","Annotation","ElementValuePair","ElementArrayValue",
    "Member","MethodDeclaration","FieldDeclaration","ConstructorDeclaration",
    "ConstantDeclaration","ArrayInitializer","VariableDeclaration",
    "LocalVariableDeclaration","VariableDeclarator","FormalParameter",
    "InferredFormalParameter","Statement","IfStatement","WhileStatement",
    "DoStatement","ForStatement","AssertStatement","BreakStatement",
    "ContinueStatement","ReturnStatement","ThrowStatement","SynchronizedStatement",
    "TryStatement","SwitchStatement","BlockStatement","StatementExpression",
    "TryResource","CatchClause","CatchClauseParameter","SwitchStatementCase",
    "ForControl","EnhancedForControl","Expression","Assignment","TernaryExpression",
    "BinaryOperation","Cast","MethodReference","LambdaExpression","Primary",
    "Literal","This","MemberReference","Invocation","ExplicitGenericInvocation",
    "InnerClassCreator","EnumConstantDeclaration","AnnotationMethod",
    "SuperMemberReference","ArraySelector","ClassReference","VoidClassReference",
    "SuperConstructorInvocation","MethodInvocation","SuperMethodInvocation",
    "ClassCreator","ArrayCreator","Selector","Modifier","Throws",
    "InterfaceMethodDeclaration","EnumBody","EnumConstant","InterfaceBody",
    "ClassBody","AnnotationElementDeclaration","ConditionalExpression",
    "FieldAccess","ArrayAccess","PostfixExpression","PrefixExpression",
    "UnaryExpression","NewExpression","QualifiedName","PackageName",
    "TypeName","ExpressionName","MethodName","AmbiguousName",
    "MultiCatch","DiamondType","WildcardType","BoundedWildcard",
    "UnboundedWildcard","Initializer","StaticInitializer","InstanceInitializer",
    "LabeledStatement","SwitchExpression","SwitchExpressionCase","YieldStatement",
    "TextBlock","RecordDeclaration","RecordComponent","PermittedSubclasses",
    "SealedModifier","PatternMatchingInstanceof","GuardedPattern",
    "NullLiteral","StringLiteral","IntegerLiteral","FloatingPointLiteral",
    "BooleanLiteral","CharacterLiteral","HexadecimalLiteral","OctalLiteral",
    "BinaryLiteral","InterfaceExtends","ClassImplements","ClassExtends",
    "BlockInitializer","EmptyStatement","ExpressionStatement","TryCatch",
    "Finally","InstanceOfExpression","ArrayExpression","PrimitiveType",
    "VoidType","GenericType","ArrayType","FunctionType","IntersectionType",
    "UnionType","ModuleDeclaration","RequiresDirective","ExportsDirective",
    "OpensDirective","UsesDirective","ProvidesDirective","AnnotationAttribute",
    "ClassType","InterfaceType","EnumType","AnnotationType","MethodType",
    "ConstructorType","FieldType","LocalVariableType","CastType","ArrayType2",
    "TypeBound","TypeBoundList","TypeArgumentList","TypeParameterList",
    "FormalParameterList","ArgumentList","VariableDeclaratorList",
    "SwitchBlockStatementGroup","SwitchLabel","ElementValueList",
    "AnnotationTypeElementDeclaration","AnnotationTypeElementDeclarator",
    "DefaultValue","EnumConstantList","InterfaceMemberDeclaration",
    "ClassMemberDeclaration","ConstructorBody","ExplicitGenericInvocationSuffix",
    "SuperSuffix","Arguments","Selector2","InnerCreator","Creator",
    "NonWildcardTypeArguments","NonWildcardTypeArgumentsOrDiamond",
    "TypeArgumentsOrDiamond","QualifiedCreator","ArrayCreatorRest",
    "ClassCreatorRest","ArrayInitializer2",
]

CPP_NODES = [
    "translation_unit","function_definition","declaration","field_declaration",
    "type_definition","struct_specifier","union_specifier","class_specifier",
    "enum_specifier","namespace_definition","using_declaration","using_directive",
    "alias_declaration","static_assert_declaration","template_declaration",
    "template_instantiation","explicit_instantiation","explicit_specialization",
    "linkage_specification","attribute_declaration","concept_definition",
    "declarator","pointer_declarator","reference_declarator","function_declarator",
    "array_declarator","abstract_declarator","abstract_pointer_declarator",
    "abstract_reference_declarator","abstract_function_declarator",
    "abstract_array_declarator","structured_binding_declarator",
    "parameter_declaration","variadic_parameter_declaration",
    "optional_parameter_declaration","compound_statement","expression_statement",
    "if_statement","switch_statement","case_statement","default_statement",
    "do_statement","while_statement","for_statement","for_range_loop",
    "return_statement","break_statement","continue_statement","goto_statement",
    "labeled_statement","null_statement","try_statement","throw_statement",
    "co_return_statement","co_yield_statement","co_await_expression",
    "expression","comma_expression","conditional_expression",
    "assignment_expression","binary_expression","unary_expression",
    "cast_expression","pointer_expression","subscript_expression",
    "call_expression","field_expression","sizeof_expression",
    "alignof_expression","offsetof_expression","new_expression",
    "delete_expression","lambda_expression","fold_expression",
    "pack_expansion_expression","requires_expression","await_expression",
    "identifier","number_literal","string_literal","char_literal",
    "concatenated_string","user_defined_literal","raw_string_literal",
    "true","false","null","nullptr","this","type_specifier","type_qualifier",
    "type_identifier","primitive_type","sized_type_specifier",
    "qualified_identifier","template_type","dependent_type",
    "placeholder_type_specifier","trailing_return_type",
    "preproc_include","preproc_def","preproc_function_def","preproc_params",
    "preproc_call","preproc_if","preproc_ifdef","preproc_ifndef",
    "preproc_else","preproc_elif","preproc_endif","preproc_defined",
    "preproc_unary_expression","preproc_binary_expression",
    "access_specifier","virtual_function","base_class_clause","friend_declaration",
    "operator_cast","operator_name","destructor_name","template_argument_list",
    "template_parameter_list","type_parameter_declaration",
    "variadic_type_parameter_declaration","initializer_list","initializer_pair",
    "argument_list","parameter_list","catch_clause","handler",
    "noexcept","throw_specifier","requires_clause","concept_requirement",
    "type_requirement","compound_requirement","nested_requirement",
    "storage_class_specifier","function_specifier","cv_qualifier",
    "ref_qualifier","exception_specification","dynamic_exception_specification",
    "ms_declspec_modifier","ms_call_modifier","ms_based_modifier",
    "ms_pointer_modifier","gnu_asm_statement","asm_string","asm_output_operand",
    "asm_input_operand","asm_clobber_list","asm_goto_list",
    "static_cast_expression","dynamic_cast_expression",
    "reinterpret_cast_expression","const_cast_expression",
    "typeid_expression","type_descriptor","char_type","char8_type",
    "char16_type","char32_type","wchar_type","bool_type","void_type",
    "auto_type","nullptr_type","dependent_name","template_method",
    "scoped_identifier","scoped_namespace_identifier","scoped_type_identifier",
    "constructor_specifier","destructor_specifier",
    "virtual_specifier","override_specifier","final_specifier",
]

STAT_FEATURES = [
    "avg_line_length",
    "whitespace_ratio",
    "empty_lines_density",
    "avg_function_length",
    "function_density",
    "avg_variable_name_length",
    "maintainability_index",
    "max_decision_op_length",
]
AST_SCALAR_FEATURES = ["ast_depth", "assignment_operators"]

PYTHON_DENSITY_FEATURES = [f"py_ast_{n}"   for n in PYTHON_NODES]
JAVA_DENSITY_FEATURES = [f"java_ast_{n}" for n in JAVA_NODES]
CPP_DENSITY_FEATURES = [f"cpp_ast_{n}"  for n in CPP_NODES]

ALL_FEATURE_NAMES = (
    STAT_FEATURES
    + AST_SCALAR_FEATURES
    + PYTHON_DENSITY_FEATURES
    + JAVA_DENSITY_FEATURES
    + CPP_DENSITY_FEATURES
)

DATASET_FEAT_MAP = {
    "avgLineLength": "avg_line_length",
    "whiteSpaceRatio": "whitespace_ratio",
    "emptyLinesDensity": "empty_lines_density",
    "avgFunctionLength": "avg_function_length",
    "functionDefinitionDensity":"function_density",
    "avgIdentifierLength": "avg_variable_name_length",
    "maintainabilityIndex": "maintainability_index",
    "maxDecisionTokens": "max_decision_op_length",
}

SOURCE_LABEL_MAP = {"cf": "Codeforces", "gh": "GitHub", "lc": "LeetCode"}

# Authorship label mapping 
# Integer class IDs used for training
MODEL_CLASS_NAMES = ["human", "llama3.1", "qwen1.5", "codellama", "nxcode", "gpt"]
NUM_CLASSES = len(MODEL_CLASS_NAMES)

# Fuzzy matching: raw model-column values → integer class
_MODEL_KEYWORDS: list[tuple[list[str], int]] = [
    (["human"], 0),
    (["llama-3.1", "llama3.1", "llama 3"], 1),
    (["qwen"], 2),
    (["codellama", "code-llama", "code_llama"], 3),
    (["nxcode", "nx-code", "nx_code"], 4),
    (["gpt"], 5),
]

def model_str_to_int(raw: str) -> int:
    """Map a raw `model` column value to an integer class id."""
    s = str(raw).strip().lower()
    for keywords, class_id in _MODEL_KEYWORDS:
        if any(kw in s for kw in keywords):
            return class_id
    raise ValueError(f"Unknown model value: '{raw}'")


# Feature extraction (unchanged from original) 

def safe_parse_python(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(str(code))
    except Exception:
        return None

def get_ast_depth(tree: Optional[ast.AST]) -> float:
    if tree is None:
        return 0.0
    def _d(node) -> int:
        ch = list(ast.iter_child_nodes(node))
        return 1 if not ch else 1 + max(_d(c) for c in ch)
    return float(_d(tree))

def count_assignments_python(tree: Optional[ast.AST]) -> int:
    if tree is None:
        return 0
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)))

def get_python_node_counts(tree: Optional[ast.AST]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if tree is None:
        return counts
    for node in ast.walk(tree):
        t = type(node).__name__
        counts[t] = counts.get(t, 0) + 1
    return counts


def get_java_node_counts(code: str) -> Dict[str, int]:
    if not JAVALANG_AVAILABLE:
        return {}
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    for i, node in tree:
        t = type(node).__name__
        counts[t] = counts.get(t, 0) + 1
    return counts

def get_java_ast_depth(code: str) -> float:
    if not JAVALANG_AVAILABLE:
        return 0.0
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return 0.0
    max_depth = [0]
    def _walk(node, depth):
        max_depth[0] = max(max_depth[0], depth)
        if hasattr(node, 'children'):
            for child in node.children:
                if isinstance(child, javalang.ast.Node):
                    _walk(child, depth + 1)
                elif isinstance(child, (list, set, frozenset)):
                    for item in child:
                        if isinstance(item, javalang.ast.Node):
                            _walk(item, depth + 1)
    _walk(tree, 0)
    return float(max_depth[0])

def count_assignments_java(code: str) -> int:
    if not JAVALANG_AVAILABLE:
        return 0
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return 0
    count = 0
    for i, node in tree:
        if isinstance(node, (
            javalang.tree.Assignment,
            javalang.tree.VariableDeclarator,
            javalang.tree.LocalVariableDeclaration,
        )):
            count += 1
    return count


def _walk_ts_tree(node):
    yield node
    for child in node.children:
        yield from _walk_ts_tree(child)

def get_cpp_node_counts(code: str) -> Dict[str, int]:
    if not TREESITTER_AVAILABLE:
        return {}
    try:
        code_bytes = code.encode("utf-8", errors="replace") if isinstance(code, str) else code
        tree = CPP_TS_PARSER.parse(code_bytes)
        counts: Dict[str, int] = {}
        for node in _walk_ts_tree(tree.root_node):
            t = node.type
            counts[t] = counts.get(t, 0) + 1
        return counts
    except Exception:
        return {}

def get_cpp_ast_depth(code: str) -> float:
    if not TREESITTER_AVAILABLE:
        return 0.0
    try:
        code_bytes = code.encode("utf-8", errors="replace") if isinstance(code, str) else code
        tree = CPP_TS_PARSER.parse(code_bytes)
        def _depth(node) -> int:
            if not node.children:
                return 1
            return 1 + max(_depth(c) for c in node.children)
        return float(_depth(tree.root_node))
    except Exception:
        return 0.0

def count_assignments_cpp(code: str) -> int:
    if not TREESITTER_AVAILABLE:
        return 0
    counts = get_cpp_node_counts(code)
    return counts.get("assignment_expression", 0)


def extract_features(code: str, precomputed: dict, language: str) -> np.ndarray:
    vec = np.full(len(ALL_FEATURE_NAMES), np.nan, dtype=np.float32)
    idx = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}

    if isinstance(precomputed, dict):
        for ds_key, feat_name in DATASET_FEAT_MAP.items():
            val = precomputed.get(ds_key, None)
            if val is not None:
                try:
                    vec[idx[feat_name]] = float(val)
                except (TypeError, ValueError):
                    pass

    if not isinstance(code, str) or not code.strip():
        return vec

    non_empty_lines = [l for l in code.splitlines() if l.strip()]
    n_lines = max(len(non_empty_lines), 1)
    lang = language.lower().strip() if isinstance(language, str) else ""

    if lang == "python":
        tree = safe_parse_python(code)
        vec[idx["ast_depth"]] = get_ast_depth(tree)
        vec[idx["assignment_operators"]] = float(count_assignments_python(tree))
        node_counts = get_python_node_counts(tree)
        for node_name in PYTHON_NODES:
            vec[idx[f"py_ast_{node_name}"]] = node_counts.get(node_name, 0) / n_lines

    elif lang == "java":
        vec[idx["ast_depth"]] = get_java_ast_depth(code)
        vec[idx["assignment_operators"]] = float(count_assignments_java(code))
        node_counts = get_java_node_counts(code)
        for java_node in JAVA_NODES:
            vec[idx[f"java_ast_{java_node}"]] = node_counts.get(java_node, 0) / n_lines

    elif lang in ("c++", "cpp", "c"):
        vec[idx["ast_depth"]] = get_cpp_ast_depth(code)
        vec[idx["assignment_operators"]] = float(count_assignments_cpp(code))
        node_counts = get_cpp_node_counts(code)
        for cpp_node in CPP_NODES:
            vec[idx[f"cpp_ast_{cpp_node}"]] = node_counts.get(cpp_node, 0) / n_lines

    return vec


def build_feature_matrix(
    codes,
    precomputed_list,
    languages,
    known_mask=None,
    missing_thresh: float = 0.20,
    verbose: bool = True,
):
    if verbose:
        print(f"  Extracting features for {len(codes)} samples ...", end="", flush=True)

    raw = np.stack([
        extract_features(c, p, l)
        for c, p, l in zip(codes, precomputed_list, languages)
    ])

    if verbose:
        print(f" done.  Shape: {raw.shape}")

    if known_mask is None:
        miss = np.isnan(raw).mean(axis=0)
        known_mask = miss <= missing_thresh
        if verbose:
            print(f"  Features retained after missing filter: {known_mask.sum()} / {len(known_mask)}")

    raw_filtered = raw[:, known_mask]
    feat_names = [n for n, k in zip(ALL_FEATURE_NAMES, known_mask) if k]
    return raw_filtered, feat_names, known_mask


# Data loading 

def load_codet_m4_authorship(
    language: str = None,
    source: str = None,
    max_train: int = None,
    max_val: int = None,
    max_test: int = None,
    seed: int = 42,
):
    """Load CoDET-M4, label by `model` column (authorship identification).

    Rows where the model value cannot be mapped are dropped with a warning.
    The function respects any pre-existing train/val/test split column in the
    dataset; otherwise it creates an 80/10/10 stratified split.
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    print(f"\nLoading CoDet-M4 (authorship mode) ...")
    ds = load_dataset("DaniilOr/CoDET-M4")
    split_key = list(ds.keys())[0]
    df = ds[split_key].to_pandas()
    print(f"  {len(df)} rows loaded")

    # Locate model column 
    model_col = next(
        (c for c in ["model", "model_name", "generator", "source_model"] if c in df.columns),
        None,
    )
    if model_col is None:
        raise ValueError(
            "Could not find a 'model' column in the dataset. "
            f"Available columns: {list(df.columns)}"
        )
    print(f"  Using '{model_col}' column for authorship labels.")
    print(f"  Unique values: {sorted(df[model_col].dropna().unique().tolist())}")

    # Map model strings → integer class ids 
    def _safe_map(v):
        try:
            return model_str_to_int(v)
        except ValueError:
            return -1  # sentinel for unknown

    df["label"] = df[model_col].map(_safe_map)
    n_unknown = (df["label"] == -1).sum()
    if n_unknown > 0:
        print(f"  WARNING: {n_unknown} rows had unrecognised model values and will be dropped.")
        df = df[df["label"] != -1].reset_index(drop=True)

    print(f"  Class distribution after mapping:")
    for cls_id, cls_name in enumerate(MODEL_CLASS_NAMES):
        n = (df["label"] == cls_id).sum()
        print(f"    {cls_id} ({cls_name}): {n}")

    # Language / source filters
    lang_col = next((c for c in ["language", "lang", "programming_language"]
                     if c in df.columns), None)
    src_col  = next((c for c in ["source", "domain", "platform"]
                     if c in df.columns), None)

    if language and lang_col:
        df = df[df[lang_col] == language].reset_index(drop=True)
    if source and src_col:
        df = df[df[src_col] == source].reset_index(drop=True)

    print(f"  After language/source filter: {len(df)} rows")

    # Split 
    split_col = next((c for c in ["split", "set", "subset"] if c in df.columns), None)
    if split_col and df[split_col].nunique() > 1:
        train_df = df[df[split_col].str.lower().isin(["train", "training"])].copy()
        val_df   = df[df[split_col].str.lower().isin(["val", "validation", "valid"])].copy()
        test_df  = df[df[split_col].str.lower().isin(["test", "testing"])].copy()
        print(f"  Pre-split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    else:
        train_df, temp = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df["label"]
        )
        val_df, test_df = train_test_split(
            temp, test_size=0.5, random_state=seed, stratify=temp["label"]
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Optional size caps 
    for split_name, split_df, limit in [
        ("train", train_df, max_train),
        ("val",  val_df,  max_val),
        ("test", test_df, max_test),
    ]:
        if limit is not None and len(split_df) > limit:
            sub, _ = train_test_split(
                split_df, train_size=limit,
                random_state=seed, stratify=split_df["label"],
            )
            if split_name == "train":   train_df = sub.reset_index(drop=True)
            elif split_name == "val":   val_df   = sub.reset_index(drop=True)
            else: test_df  = sub.reset_index(drop=True)

    print(f"  Final — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    for split_df in (train_df, val_df, test_df):
        split_df["_lang"] = split_df[lang_col] if lang_col else language or "unknown"
        split_df["_src"]  = split_df[src_col]  if src_col  else "unknown"

    return train_df, val_df, test_df


# Metrics

def compute_metrics(y_true, y_pred) -> dict:
    """Macro-averaged multiclass metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average="macro"),
        "recall":  recall_score(y_true, y_pred, zero_division=0, average="macro"),
        "f1":  f1_score(y_true, y_pred, zero_division=0, average="macro"),
        "n":  int(len(y_true)),
    }


def print_metrics(y_true, y_pred, title="Test"):
    res = compute_metrics(y_true, y_pred)
    # Only show target names for classes that are present in this split
    present_ids   = sorted(set(y_true) | set(y_pred))
    present_names = [MODEL_CLASS_NAMES[i] for i in present_ids]

    print(f"\n  {title}")
    print(f"  Accuracy : {res['accuracy']:.4f}   Precision: {res['precision']:.4f}"
          f"   Recall: {res['recall']:.4f}   F1: {res['f1']:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels=present_ids, target_names=present_names,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=present_ids)
    print("  Confusion matrix (rows=true, cols=pred):")
    header = "".join(f"{name:>12}" for name in present_names)
    print(f"  {'':>12}{header}")
    for i, name in enumerate(present_names):
        row = "".join(f"{cm[i, j]:>12d}" for j in range(len(present_names)))
        print(f"  {name:>12}{row}")
    return res


# Entry point

if __name__ == "__main__":
    try:
        from catboost import CatBoostClassifier, Pool
        CATBOOST_OK = True
    except ImportError:
        CATBOOST_OK = False

    if not CATBOOST_OK:
        raise SystemExit("Install catboost first:  pip install catboost -q")
    if not JAVALANG_AVAILABLE:
        print("javalang not installed. Java AST features will be zeros.")
    if not TREESITTER_AVAILABLE:
        print("tree-sitter not installed. C++ AST features will be zeros.")

    def _has_cuda() -> bool:
        return (shutil.which("nvidia-smi") is not None and
                subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0)

    TASK_TYPE = "GPU" if _has_cuda() else "CPU"
    print(f"  CatBoost task_type: {TASK_TYPE}")

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        SAVE_DIR = "/content/drive/MyDrive/codet_m4_catboost_authorship"
    except Exception:
        SAVE_DIR = "./codet_m4_catboost_authorship"

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"  Saving to: {SAVE_DIR}")

    LANGUAGES_TO_RUN = ["python", "java", "cpp"]
    SOURCE = None
    MAX_TRAIN  = None
    MAX_VAL = None
    MAX_TEST = None
    N_TREES = 2_000
    LEARNING_RATE = 0.1

    KNOWN_SOURCES = ["cf", "gh", "lc"]

    all_results = {}
    all_src_results = {}

    for LANGUAGE in LANGUAGES_TO_RUN:
        print(f"\n  LANGUAGE: {LANGUAGE}")

        lang_dir = os.path.join(SAVE_DIR, LANGUAGE.replace("+", "pp"))
        os.makedirs(lang_dir, exist_ok=True)

        train_df, val_df, test_df = load_codet_m4_authorship(
            language=LANGUAGE, source=SOURCE,
            max_train=MAX_TRAIN, max_val=MAX_VAL, max_test=MAX_TEST,
        )

        code_col = "cleaned_code" if "cleaned_code" in train_df.columns else "code"
        feat_col = "features"  if "features"  in train_df.columns else None

        def _precomp(df): return df[feat_col].tolist() if feat_col else [{}] * len(df)
        def _langs(df):   return df["_lang"].tolist()

        print(f"\n  Feature extraction — {LANGUAGE}")
        X_train, feat_names, feat_mask = build_feature_matrix(
            train_df[code_col].tolist(), _precomp(train_df), _langs(train_df))
        y_train = train_df["label"].values.astype(int)

        X_val, _, _ = build_feature_matrix(
            val_df[code_col].tolist(), _precomp(val_df), _langs(val_df),
            known_mask=feat_mask)
        y_val = val_df["label"].values.astype(int)

        X_test, _, _ = build_feature_matrix(
            test_df[code_col].tolist(), _precomp(test_df), _langs(test_df),
            known_mask=feat_mask)
        y_test = test_df["label"].values.astype(int)

        with open(os.path.join(lang_dir, "feature_names.json"), "w") as fh:
            json.dump(feat_names, fh, indent=2)

        # CatBoost: MultiClass mode 
        print(f"\n  Training CatBoost (MultiClass) — {LANGUAGE}")
        model = CatBoostClassifier(
            iterations=N_TREES,
            learning_rate=LEARNING_RATE,
            depth=6,
            loss_function="MultiClass",   # ← multiclass cross-entropy
            eval_metric="TotalF1",        # ← macro F1 across all classes
            classes_count=NUM_CLASSES,
            task_type=TASK_TYPE,
            devices="0",
            random_seed=42,
            verbose=200,
            early_stopping_rounds=100,
        )

        train_pool = Pool(X_train, y_train, feature_names=feat_names)
        val_pool = Pool(X_val,   y_val,   feature_names=feat_names)
        model.fit(train_pool, eval_set=val_pool)
        print(f"  Best iteration: {model.best_iteration_}")

        model_path = os.path.join(lang_dir, f"catboost_authorship_{LANGUAGE.replace('+','p')}.cbm")
        model.save_model(model_path)

        # Training curves 
        evals = model.get_evals_result()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"CatBoost Authorship Training Curves — {LANGUAGE}", fontsize=14)
        for ax, metric in zip(axes, ["MultiClass", "TotalF1"]):
            train_vals = evals.get("learn", {}).get(metric, [])
            val_vals   = evals.get("validation", {}).get(metric, [])
            if train_vals:
                ax.plot(range(1, len(train_vals) + 1), train_vals, label="Train", linewidth=1.2)
            if val_vals:
                ax.plot(range(1, len(val_vals) + 1), val_vals, label="Validation", linewidth=1.2)
            if model.best_iteration_ is not None and val_vals:
                best_val = val_vals[model.best_iteration_]
                ax.axvline(model.best_iteration_ + 1, color="red", linestyle="--",
                           linewidth=1, label=f"Best iter ({model.best_iteration_+1})")
                ax.scatter([model.best_iteration_ + 1], [best_val], color="red", zorder=5)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lang_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        with open(os.path.join(lang_dir, "training_curves.json"), "w") as fh:
            json.dump(evals, fh, indent=2)

        # Evaluation
        print(f"\n  Evaluation — {LANGUAGE}")
        y_pred  = model.predict(X_test).flatten().astype(int)
        results = print_metrics(y_test, y_pred, title=f"Test Results — {LANGUAGE}")
        all_results[LANGUAGE] = results

        # Per-class breakdown plot
        per_class_f1 = f1_score(y_test, y_pred, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(MODEL_CLASS_NAMES, per_class_f1, color="steelblue")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
        ax.set_title(f"Per-Class F1 — CatBoost Authorship — {LANGUAGE}")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lang_dir, "per_class_f1.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Feature importances 
        importances = model.get_feature_importance(train_pool)
        top_idx = np.argsort(importances)[::-1][:20]
        importance_df = pd.DataFrame({
            "feature":    feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df.to_csv(os.path.join(lang_dir, "feature_importances.csv"), index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        top_names  = [feat_names[i]  for i in top_idx]
        top_scores = [importances[i] for i in top_idx]
        ax.barh(top_names[::-1], top_scores[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-20 Feature Importances — {LANGUAGE}")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lang_dir, "feature_importances_top20.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Per-source breakdown 
        src_results = {}
        print(f"\n  Per-source breakdown — {LANGUAGE}")
        print(f"  {'Source':<14} {'P':>7} {'R':>7} {'F1':>7} {'Acc':>7}  {'n':>5}")
        for src in KNOWN_SOURCES:
            mask = (test_df["_src"] == src).values
            if not mask.any():
                continue
            yt, yp = y_test[mask], y_pred[mask]
            src_res = compute_metrics(yt, yp)
            src_results[src] = src_res
            label = SOURCE_LABEL_MAP.get(src, src)
            print(f"  {label:<14} "
                  f"{src_res['precision']:7.4f} "
                  f"{src_res['recall']:7.4f} "
                  f"{src_res['f1']:7.4f} "
                  f"{src_res['accuracy']:7.4f}  "
                  f"{src_res['n']:5d}")
        all_src_results[LANGUAGE] = src_results

        lang_metrics = {
            "language": LANGUAGE,
            "task": "authorship_identification",
            "classes": MODEL_CLASS_NAMES,
            "best_iteration": int(model.best_iteration_) if model.best_iteration_ is not None else N_TREES,
            "n_features": len(feat_names),
            "overall": results,
            "per_source": src_results,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        }
        with open(os.path.join(lang_dir, "metrics.json"), "w") as fh:
            json.dump(lang_metrics, fh, indent=2)

    # Cross-language summary
    print(f"\n\n  CROSS-LANGUAGE SUMMARY")
    print(f"  {'Language':<10} {'P':>7} {'R':>7} {'F1':>7} {'Acc':>7}  {'n':>6}")
    for lang, res in all_results.items():
        print(f"  {lang:<10} "
              f"{res['precision']:7.4f} "
              f"{res['recall']:7.4f} "
              f"{res['f1']:7.4f} "
              f"{res['accuracy']:7.4f}  "
              f"{res['n']:6d}")

    # Summary bar chart 
    langs = list(all_results.keys())
    metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(langs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics_to_plot):
        vals = [all_results[l][metric] for l in langs]
        ax.bar(x + i * width, vals, width, label=metric.capitalize())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(langs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CatBoost Authorship — Per-Language Results")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "summary_all_languages.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "task": "authorship_identification",
        "classes": MODEL_CLASS_NAMES,
        "per_language": all_results,
        "per_source_per_lang": all_src_results,
    }
    with open(os.path.join(SAVE_DIR, "summary_all.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n  Done.")