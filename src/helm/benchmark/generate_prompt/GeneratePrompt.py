import os
from typing import List, Optional
from dataclasses import dataclass, replace
from helm.benchmark.run_spec import RunSpec
from helm.benchmark.executor import Executor
from helm.benchmark.scenarios.scenario import Instance
from helm.benchmark.adaptation.adapters.adapter_factory import AdapterFactory
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.common.hierarchical_logger import hlog
from helm.benchmark.window_services.tokenizer_service import TokenizerService

@dataclass
class TextInput:
    text: str

@dataclass
class GenerationResult:
    """Result of a prompt generation."""
    prompt_generated: str
    run_spec_name: str
    model: str

class GeneratePrompt:
    """
    Class for generating prompts using LLMs based on run specs.
    """
    def __init__(self, description: str, run_spec: List[RunSpec], executor: Executor):
        self.description = description
        self.run_specs = run_spec
        self.executor = executor
        
        # Creating a TokenizerService
        if hasattr(executor, 'execution_spec') and hasattr(executor.execution_spec, 'auth'):
            self.tokenizer_service = TokenizerService(executor.service, executor.execution_spec.auth)
        else:
            hlog("Warning: Could not create a tokenizer_service. Some features may be limited.")
            self.tokenizer_service = None
    
    def generate(self) -> List[GenerationResult]:
        """Generate prompts for each run spec."""
        results = []
        for run_spec in self.run_specs:
            try:
                result = self.generate_for_spec(run_spec)
                if result:
                    results.append(result)
            except Exception as e:
                hlog(f"Error generating prompt with run_spec {run_spec.name}: {str(e)}")
        
        return results
    
    def generate_for_spec(self, run_spec: RunSpec) -> Optional[GenerationResult]:
        """Generate prompt for a single run spec."""
        # Extract model information from run_spec
        model = self._extract_model_from_run_spec(run_spec)
        if not model:
            raise ValueError(f"Could not extract model from run_spec {run_spec.name}")
        
        arq_description = self._load_description()
        # Create test instance with the correct input format
        instance = Instance(
            id="prompt_generation",
            input=TextInput(text=f"Gere um prompt para essa descrição: {arq_description}, o prompt vai ser usado para avaliar o benchamrak da descrição"),
            references=[],
            split="test"
        )
        
        # Modify adapter_spec
        modified_adapter_spec = replace(run_spec.adapter_spec, 
            method='generation',
            instructions=f"Você é especialista em avaliação de modelos de linguagem. Crie um prompt detalhado para avaliar {self.description} no contexto de benchmarks de modelos de linguagem. O prompt deve ser específico, claro e permitir uma avaliação objetiva das capacidades do modelo.",
            input_prefix='', 
            output_prefix='', 
            max_tokens=200,
            stop_sequences=[],
            temperature=0.7
        )
        
        # Adapt the main instance
        adapter = AdapterFactory.get_adapter(modified_adapter_spec, self.tokenizer_service)
        request_states = adapter.adapt([instance], 1)
        
        # Create scenario_state with the modified adapter_spec
        scenario_state = ScenarioState(
            adapter_spec=modified_adapter_spec,
            request_states=request_states,
            annotator_specs=[]
        )
        
        # Update the request to match the modified settings
        for rs in scenario_state.request_states:
            modified_request = replace(rs.request,
                temperature=0.7,
                max_tokens=200,
                stop_sequences=[]
            )
            scenario_state.request_states[0] = replace(rs, request=modified_request)
        
        # Execute
        scenario_state = self.executor.execute(scenario_state)
        
        # Process result
        if len(scenario_state.request_states) > 0:
            request_state = scenario_state.request_states[0]
            
            if request_state.result and hasattr(request_state.result, "completions"):
                if request_state.result.completions:
                    completion = request_state.result.completions[0]
                    
                    if hasattr(completion, 'text'):
                        generated_text = completion.text
                    else:
                        generated_text = str(completion)
                    
                    return GenerationResult(
                        prompt_generated=generated_text,
                        run_spec_name=run_spec.name,
                        model=model
                    )
    
    def _extract_model_from_run_spec(self, run_spec: RunSpec) -> Optional[str]:
        """Extract model name from run_spec."""
        # Get model directly from adapter_spec
        if hasattr(run_spec.adapter_spec, 'model'):
            return run_spec.adapter_spec.model
            
        # Extract from run_spec name (common format: name:model=X,groups=Y)
        if ':model=' in run_spec.name:
            parts = run_spec.name.split(':')
            if len(parts) > 1:
                params = parts[1].split(',')
                for param in params:
                    if param.startswith('model='):
                        return param.split('=')[1]
        
        # Get from other properties
        if hasattr(run_spec, 'args') and 'model' in run_spec.args:
            return run_spec.args['model']
        
        # Check nested fields
        if hasattr(run_spec.adapter_spec, 'config') and hasattr(run_spec.adapter_spec.config, 'model'):
            return run_spec.adapter_spec.config.model

    def _load_description(self):
        """
        Loads the contents of the description file in txt format.
        """
        base_path = "src/helm/benchmark/generate_prompt/descriptions"
        file_path = os.path.join(base_path, f"{self.description}.txt")
            
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            alt_path = os.path.join("descriptions", f"{self.description}.txt")
            try:
                with open(alt_path, 'r') as file:
                    return file.read()
            except FileNotFoundError:
                return f"Generate a prompt for: {self.description}"
        except Exception as e:
            hlog(f"Error reading file {file_path}: {str(e)}")
            return f"Generate a prompt for: {self.description}"