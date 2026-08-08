from dotenv import load_dotenv
from pathlib import Path
from deepeval.dataset.dataset import EvaluationDataset
from app.rag_workflow import graph
from deepeval.test_case.llm_test_case import LLMTestCase
from config.parameter_config import params_config

# load evaluation dataset params
evaluation_dataset_params=params_config.evaluation_dataset
golden_dataset_params  = params_config.golden_dataset

#load the api keys
load_dotenv()

def generate_evaluation_dataset():

    # create the paths
    ROOT_DIR =Path(__file__).resolve().parent.parent.parent

    GOLDENS_PATH = (ROOT_DIR/"data"/"evaluation"/"goldens"/golden_dataset_params.golden_dataset_filename).with_suffix(".json")
    EVALUATION_DATA_DIR = ROOT_DIR / "data" / "evaluation" / "eval_dataset"

    #create dir
    EVALUATION_DATA_DIR.mkdir(exist_ok=True , parents=True)

    #dataset to read golden from

    golden_dataset =EvaluationDataset()
    golden_dataset.add_goldens_from_json_file(file_path=GOLDENS_PATH)

    #dataset to hold produced test cases
    eval_dataset = EvaluationDataset()

    for golden in golden_dataset.goldens:
        final_state = graph.invoke({"query": golden.input})
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=final_state.get("response"),
            expected_output=golden.expected_output,
            retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
        )
        eval_dataset.add_test_case(test_case=test_case)
        
    eval_dataset.save_as(
        file_name=evaluation_dataset_params.evaluation_dataset_filename,
        file_type="json",
        directory=EVALUATION_DATA_DIR,
        include_test_cases=True
    )
    
if __name__ =="__main__":
    generate_evaluation_dataset()
        

