# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import asyncio
import logging
import uuid

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TypeVar, Self, Union, overload

from treelib.tree import Node, Tree

from pyrit.common.path import DATASETS_PATH
from pyrit.common.utils import combine_dict, warn_if_set
from pyrit.exceptions import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)
from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackContext,
    AttackConverterConfig,
    AttackScoringConfig,
    AttackStrategy,
    AttackStrategyResultT,
)
from pyrit.memory import CentralMemory
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ConversationReference,
    ConversationType,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import (
    Scorer,
    SelfAskScaleScorer,
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
)

logger = logging.getLogger(__name__)

MultiBranchAttackContextT = TypeVar("MultiBranchAttackContextT", bound="MultiBranchAttackContext")
CmdT = TypeVar("CmdT", bound="MultiBranchCommand")

class MultiBranchCommand(Enum):
    """
    All possible commands that can be executed in a multi-branch attack.
    You can think of this as the possible states of the attack object, where
    the handler is a transition function between states.
    
    """
    CONTINUE = "continue" # Send a new prompt to continue the conversation
    UP = "up" # Move to parent node
    DOWN = "down" # Move to a child node (if multiple children, specify which.
    # (if no children are specified, create a new one)
    END = "end" # End the attack

@dataclass
class ConversationNode(Node):
    """
    A node in the conversation tree representing a conversation state.
    
    Each node tracks its position in the conversation tree and can have
    multiple children representing different conversation branches.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_node: Optional['ConversationNode'] = None
    children_nodes: List['ConversationNode'] = field(default_factory=list)
    depth: int = 0
    turn_count: int = 0
    last_user_prompt: Optional[str] = None
    last_target_response: Optional[PromptRequestPiece] = None
    scores: Dict[str, Score] = field(default_factory=dict)
    
    def add_child(self, *, child: 'ConversationNode') -> None:
        """Add a child node to this conversation node."""
        child.parent_node = self
        child.depth = self.depth + 1
        self.children_nodes.append(child)
    
    def get_siblings(self) -> List['ConversationNode']:
        """Get all sibling nodes (children of the same parent)."""
        if not self.parent_node:
            return []
        return [child for child in self.parent_node.children_nodes if child.node_id != self.node_id]
    
    def get_path_from_root(self) -> List['ConversationNode']:
        """Get the path from root to this node."""
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent_node
        return path

@dataclass
class MultiBranchAttackContext(AttackContext):
    """
    Context for multi-branch attacks.
    
    Parameters 
    """

    # Tree structure to hold the branches of the attack
    attack_tree: Tree = field(default_factory=lambda: Tree(node_class=ConversationNode))

    # Conversation tree state
    root_node: Optional[ConversationNode] = None
    current_node: Optional[ConversationNode] = None
    all_nodes: List[ConversationNode] = field(default_factory=list)
    
    # Execution tracking
    total_turns: int = 0
    total_branches: int = 0
    
@dataclass
class MultiBranchAttackResult(AttackResult):
    """Result of multi-branch attack execution."""

    # Conversation tree metrics
    total_turns: int = 0
    total_branches: int = 0
    max_depth_reached: int = 0
    final_node_id: str = ""
    
    # Tree structure for analysis, maybe visualize similar to what TAP did
    conversation_tree: Optional[Dict] = None

    def display_conversation_tree(self) -> None:
        """
        Display the conversation tree in a readable format.
        
        Shows the complete tree structure with prompts, responses, and scores
        using proper indentation to represent the hierarchy.
        """
        if not self.conversation_tree:
            print("\nNo conversation tree available to display.")
            return
        
        print(f"\n{'='*80}")
        print("CONVERSATION TREE SUMMARY")
        print(f"Total Turns: {self.total_turns} | Total Branches: {self.total_branches} | Max Depth: {self.max_depth_reached}")
        print(f"{'='*80}")
        
        self._display_tree_node_recursive(node=self.conversation_tree, indent=0)
        print(f"{'='*80}")

    def _display_tree_node_recursive(self, *, node: Dict, indent: int) -> None:
        """
        Recursively display a tree node and its children with proper indentation.
        
        Args:
            node (Dict): The tree node dictionary to display.
            indent (int): Current indentation level.
        """
        # Create indentation string
        indent_str = "    " * indent
        branch_str = "└── " if indent > 0 else ""
        
        # Node header
        if node.get("depth", 0) == 0:
            print(f"{indent_str}ROOT (Turn {node.get('turn_count', 0)})")
        else:
            print(f"{indent_str}{branch_str}Node {node.get('depth', 0)} (Turn {node.get('turn_count', 0)})")
        
        # Display prompt if available
        last_prompt = node.get("last_prompt")
        if last_prompt:
            prompt_preview = last_prompt[:100] + ("..." if len(last_prompt) > 100 else "")
            print(f"{indent_str}    USER: {prompt_preview}")
        
        # Display response if available
        response = node.get("response")
        if response:
            # Handle both PromptRequestPiece objects and string representations
            if hasattr(response, 'converted_value'):
                response_content = response.converted_value
            elif isinstance(response, dict) and 'converted_value' in response:
                response_content = response['converted_value']
            elif isinstance(response, str):
                response_content = response
            else:
                response_content = str(response)
            
            response_preview = response_content[:100] + ("..." if len(response_content) > 100 else "")
            print(f"{indent_str}    TARGET: {response_preview}")
        
        # Display scores if available
        scores = node.get("scores", {})
        if scores:
            scores_str = ", ".join([f"{name}={value}" for name, value in scores.items()])
            print(f"{indent_str}    SCORES: {scores_str}")
        
        # Display node ID (shortened)
        node_id = node.get("node_id", "")
        if node_id:
            node_id_short = node_id[:8] + "..." if len(node_id) > 8 else node_id
            print(f"{indent_str}    NODE_ID: {node_id_short}")
        
        # Add spacing between nodes at same level
        if indent > 0:
            print()
        
        # Recursively display children
        children = node.get("children", [])
        for i, child in enumerate(children):
            if i > 0:
                print()  # Add space between siblings
            self._display_tree_node_recursive(node=child, indent=indent + 1)   


class MultiBranchAttack(AttackStrategy[MultiBranchAttackContext, MultiBranchAttackResult], ABC):
    """
    Interactive human-guided multi-turn attack with conversation tree navigation.
    
    This attack allows a human operator to manually conduct multi-turn conversations
    with a target system while maintaining a tree structure of conversation states.
    The operator can branch conversations, backtrack to previous states, and explore
    different conversation paths.
    
    Key features:
    - Manual prompt input with real-time target responses
    - Conversation tree with backtracking capabilities
    - Navigate up/down conversation branches
    - Optional scoring at each turn
    - Visual conversation history and navigation aids
    """

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        max_turns_per_branch: int = 10,
        show_conversation_history: bool = True,
    ):
        """
        Initialize the human-guided Multi-Branch Attack.

        Args:
            objective_target (PromptChatTarget): The target system to attack.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring.
            prompt_normalizer (Optional[PromptNormalizer]): Prompt normalizer to use.
            max_turns_per_branch (int): Maximum turns allowed per conversation branch.
            show_conversation_history (bool): Whether to show conversation history during interaction.
        
        Raises:
            ValueError: If objective_target is not a PromptChatTarget or parameters are invalid.
        """
        if max_turns_per_branch < 1:
            raise ValueError("max_turns_per_branch must be at least 1.")
        
        # Initialize base class
        super().__init__(logger=logger, context_type=MultiBranchAttackContext)
        
        self._memory = CentralMemory.get_memory_instance()
        
        # Store target configuration
        self._objective_target = objective_target
        if not isinstance(self._objective_target, PromptChatTarget):
            raise ValueError("The objective target must be a PromptChatTarget for human multi-turn attack.")
        
        # Initialize converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters
        
        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()
        self._objective_scorer = attack_scoring_config.objective_scorer
        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers or []
        self._successful_objective_threshold = attack_scoring_config.successful_objective_threshold
        
        # Store execution configuration
        self._max_turns_per_branch = max_turns_per_branch
        self._show_conversation_history = show_conversation_history
        
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()

    def _validate_context(self, *, context: MultiBranchAttackContext):
        if not context.objective:
            raise ValueError("The attack objective must be set in the context.")

    async def _setup_async(self, *, context: MultiBranchAttackContext) -> None:
        """
        Setup phase before executing the attack.
        
        Args:
            context (MultiBranchAttackContext): The attack context.
        """
        # Update memory labels for this execution
        context.memory_labels = combine_dict(existing_dict=self._memory_labels, new_dict=context.memory_labels)
        
        # Initialize conversation tree with root node
        context.root_node = ConversationNode(depth=0, turn_count=0)
        context.current_node = context.root_node
        context.all_nodes = [context.root_node]
        
        # Initialize tracking
        context.total_turns = 0
        context.total_branches = 0
        
        self._logger.info("Human Multi-Turn Attack initialized")
        self._logger.info(f"Objective: {context.objective}")
        self._logger.info(f"Max turns per branch: {self._max_turns_per_branch}")
    
    
    def _teardown_async(self, *, context):
        """
        Free all memory and make sure result object is well-placed
        """
        return super()._teardown_async(context=context)

    async def _perform_async(self, *, context: MultiBranchAttackContext) -> MultiBranchAttackResult:
        """
        Execute the human multi-turn attack.
        
        Args:
            context (MultiBranchAttackContext): The attack context.

        Returns:
            MultiBranchAttackResult: The result of the attack execution.
        """
        self._logger.info("Starting interactive multi-turn conversation")
        
        # Display initial instructions
        self._display_instructions()
        
        # Main conversation loop
        while True:
            try:
                # Show current state
                await self._display_current_state(context)
                
                # Get user's next action
                choice = await self._get_user_choice()
                
                if choice == MultiBranchCommand.END:
                    break
                elif choice == MultiBranchCommand.CONTINUE:
                    objective_achieved = await self._handle_continue_conversation(context)
                    # Check if objective was achieved and exit if so
                    if objective_achieved:
                        break
                elif choice == MultiBranchCommand.UP:
                    await self._handle_navigate_up(context)
                elif choice == MultiBranchCommand.DOWN:
                    await self._handle_navigate_down(context)
                
            except KeyboardInterrupt:
                print("\nAttack interrupted by user.")
                break
            except Exception as e:
                self._logger.error(f"Error during conversation: {e}")
                print(f"Error occurred: {e}")
                continue
        
        await self._display_current_state(context)
        return self._create_attack_result(context)
    

    def _display_instructions(self) -> None:
        instructions = """
        ╭─────────────────────────────────────────────────────────────────────────────╮
        │                       Human Multi-Branch Attack                             │
        ├─────────────────────────────────────────────────────────────────────────────┤
        │ Commands:                                                                   │
        │   continue - Send a new prompt to continue the conversation                 │
        │   up       - Go back to parent conversation state (backtrack)               │
        │   down     - Navigate to a child conversation branch                        │
        │   end      - End the attack and return results                              │
        │                                                                             │
        │ Navigation:                                                                 │
        │   • Each prompt creates a new conversation state                            │
        │   • You can backtrack to any previous state and branch from there           │
        │   • Multiple branches can be created from the same conversation state       │
        ╰─────────────────────────────────────────────────────────────────────────────╯
        """
        print(instructions)
    
    async def _display_current_state(self, context: MultiBranchAttackContext) -> None:
        """
        Display the current conversation state.
        
        Args:
            context (MultiBranchAttackContext): The attack context.
        """
        current = context.current_node
        if not current:
            raise ValueError("Current node is not set in the context.")
        
        print(f"\n{'='*80}")
        print(f"Current Position: Depth {current.depth}, Turn {current.turn_count}")
        print(f"Node ID: {current.node_id}")
        print(f"Conversation ID: {current.conversation_id}")
        print(f"Total Turns in Attack: {context.total_turns}")
        print(f"Total Branches Created: {context.total_branches}")
        
        # Show conversation path
        path = current.get_path_from_root()
        if len(path) > 1:
            print(f"\nConversation Path: {' → '.join([f'Node-{i}' for i in range(len(path))])}")
        
        # Show navigation options
        nav_info = []
        if current.parent_node:
            nav_info.append("UP available")
        if current.children_nodes:
            nav_info.append(f"DOWN available ({len(current.children_nodes)} branches)")
        if current.turn_count < self._max_turns_per_branch:
            nav_info.append("CONTINUE available")
        
        if nav_info:
            print(f"Navigation: {' | '.join(nav_info)}")
        
        # Show conversation history for current branch
        if self._show_conversation_history and current.conversation_id:
            await self._display_conversation_history(current.conversation_id)
        
        print(f"{'='*80}")
    
    async def _display_conversation_history(self, conversation_id: str) -> None:
        """
        Display the conversation history for a given conversation ID.
        
        Args:
            conversation_id (str): The conversation ID to display.
        """
        try:
            messages = self._memory.get_conversation(conversation_id=conversation_id)
            
            if messages:
                print(f"\nConversation History ({len(messages)} messages):")
                print("-" * 50)
                
                for idx, message in enumerate(messages, 1):
                    piece = message.get_piece()
                    role_display = piece.role.upper()
                    content = piece.converted_value[:200] + ("..." if len(piece.converted_value) > 200 else "")
                    # Color coding based on role
                    if piece.role == "user":
                        color = "\033[94m"  # blue
                    elif piece.role == "assistant":
                        color = "\033[93m"  # yellow
                    else:
                        color = "\033[95m"  # magenta
                    
                    print(f"{color}{idx}. {role_display}: {content}\033[0m")
                
                print("-" * 50)
            else:
                print("\nNo conversation history yet.")
                
        except Exception as e:
            self._logger.warning(f"Could not display conversation history: {e}")
    
    async def _get_user_choice(self) -> MultiBranchCommand:
        """
        Get the user's choice for next action.
        
        Returns:
            MultiBranchCommand: The user's selected action.
        """
        while True:
            try:
                choice_input = input("\nChoose action [[C]ontinue/[U]p/[D]own/[E]nd]: ")
                choice_input = choice_input.strip().lower()
                
                if choice_input in ["c", "continue"]:
                    return MultiBranchCommand.CONTINUE
                elif choice_input in ["u", "up"]:
                    return MultiBranchCommand.UP
                elif choice_input in ["d", "down"]:
                    return MultiBranchCommand.DOWN
                elif choice_input in ["e", "end"]:
                    return MultiBranchCommand.END
                else:
                    print("Invalid choice. Please enter: continue, up, down, or end")
                    
            except (EOFError, KeyboardInterrupt):
                return MultiBranchCommand.END
    
    async def _handle_continue_conversation(self, context: MultiBranchAttackContext) -> bool:
        """
        Handle continuing the conversation with a new prompt.
        
        Args:
            context (MultiBranchAttackContext): The attack context.
            
        Returns:
            bool: True if objective was achieved, False otherwise.
        """
        current = context.current_node

        if not current:
            raise ValueError("Current node is not set in the context.")
        
        # Check turn limit
        if current.turn_count >= self._max_turns_per_branch:
            print(f"Maximum turns ({self._max_turns_per_branch}) reached for this branch.")
            print("Use 'up' to backtrack or 'end' to finish the attack.")
            return False
        
        # Get user's prompt
        user_prompt = input("\nEnter your prompt: ")
        if not user_prompt.strip():
            print("Empty prompt. Please enter a valid prompt.")
            return False
        
        try:
            # Create a new child node for this conversation continuation
            new_node = ConversationNode(
                depth=current.depth + 1,
                turn_count=current.turn_count + 1,
                last_user_prompt=user_prompt
            )
            
            # Set up conversation ID - either duplicate parent's conversation or create new
            if current.conversation_id and current.turn_count > 0:
                # Duplicate the existing conversation to preserve history
                new_node.conversation_id = self._memory.duplicate_conversation_turns(
                    conversation_id=current.conversation_id,
                    turns=current.turn_count,
                )
            else:
                # First turn, create new conversation
                new_node.conversation_id = str(uuid.uuid4())
            
            # Add to tree structure
            current.add_child(child=new_node)
            context.all_nodes.append(new_node)
            context.current_node = new_node
            context.total_branches += 1
            
            # Send prompt to target
            print(f"\nSending prompt to target...")
            response = await self._send_prompt_to_target_async(
                prompt=user_prompt,
                conversation_id=new_node.conversation_id,
                context=context
            )
            
            # Store response in node
            new_node.last_target_response = response.get_piece()
            
            # Display response
            response_content = response.get_piece().converted_value
            print(f"\nTarget Response:")
            print("-" * 50)
            print(response_content)
            print("-" * 50)
            
            # Score response if scorer is available and check for objective achievement
            objective_achieved = False
            if self._objective_scorer:
                objective_achieved = await self._score_response_async(node=new_node, objective=context.objective)
            
            # Update counters
            context.total_turns += 1
            
            # Check if objective is achieved - if so, end attack immediately
            if objective_achieved:
                context.current_node = new_node  # Ensure final node is set
                return True  # Return success to end the attack
                
            return False  # Continue the attack
            
        except Exception as e:
            self._logger.error(f"Error sending prompt: {e}")
            print(f"Error occurred while sending prompt: {e}")
            return False
    
    async def _handle_navigate_up(self, context: MultiBranchAttackContext) -> None:
        """
        Handle navigating up to parent conversation state.
        
        Args:
            context (MultiBranchAttackContext): The attack context.
        """
        current = context.current_node

        if not current:
            raise ValueError("Current node is not set in the context.")

        if not current.parent_node:
            print("Already at root level. Cannot navigate up further.")
            return
        
        context.current_node = current.parent_node
        print(f"Moved up to parent node (Depth {context.current_node.depth})")
    
    async def _handle_navigate_down(self, context: MultiBranchAttackContext) -> None:
        """
        Handle navigating down to child conversation branches.
        
        Args:
            context (HumanMultiTurnContext): The attack context.
        """
        current = context.current_node
        if not current:
            raise ValueError("Current node is not set in the context.")
        if not current.children_nodes:
            print("No child branches available. Use 'continue' to create a new branch.")
            return
        
        if len(current.children_nodes) == 1:
            # Only one child, navigate directly
            context.current_node = current.children_nodes[0]
            print(f"Moved down to child node (Depth {context.current_node.depth})")
        else:
            # Multiple children, let user choose
            print(f"\nAvailable branches ({len(current.children_nodes)}):")
            for idx, child in enumerate(current.children_nodes):
                last_prompt = "<no prompt>"
                if child.last_user_prompt:
                    last_prompt = child.last_user_prompt[:50] + "..." if len(child.last_user_prompt) > 50 else child.last_user_prompt
                print(f"{idx + 1}. Depth {child.depth}, Prompt: '{last_prompt}'")
            
            try:
                choice_input = input(f"Choose branch [1-{len(current.children_nodes)}]: ")
                choice_idx = int(choice_input) - 1
                
                if 0 <= choice_idx < len(current.children_nodes):
                    context.current_node = current.children_nodes[choice_idx]
                    print(f"Moved down to branch {choice_idx + 1} (Depth {context.current_node.depth})")
                else:
                    print("Invalid branch number.")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("Invalid input. Staying at current node.")
    
    async def _send_prompt_to_target_async(
        self, 
        *, 
        prompt: str, 
        conversation_id: str, 
        context: MultiBranchAttackContext
    ) -> PromptRequestResponse:
        """
        Send a prompt to the objective target.
        
        Args:
            prompt (str): The prompt to send.
            conversation_id (str): The conversation ID to use.
            context (MultiBranchAttackContext): The attack context.
            
        Returns:
            PromptRequestResponse: The response from the target.
        """
        seed_prompt_group = SeedPromptGroup(
            prompts=[SeedPrompt(value=prompt, data_type="text")]
        )
        
        response = await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            target=self._objective_target,
            conversation_id=conversation_id,
            orchestrator_identifier=self.get_identifier(),
            labels=context.memory_labels,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters
        )
        
        return response
    
    async def _score_response_async(self, *, node: ConversationNode, objective: str) -> bool:
        """
        Score a response using the configured scorers, aligned with TreeOfAttacks implementation.
        
        Args:
            node (ConversationNode): The conversation node to score.
            objective (str): The objective to score against.
            
        Returns:
            bool: True if objective is achieved (score meets threshold), False otherwise.
        """
        if not node.last_target_response:
            return False
        objective_achieved = False
        try:
            # Create PromptRequestResponse from the response piece for scoring
            response_for_scoring = PromptRequestResponse(
                request_pieces=[node.last_target_response]
            )
            
            # Use the Scorer utility method to handle all scoring (same as TreeOfAttacks)
            from pyrit.score import Scorer
            scoring_results = await Scorer.score_response_with_objective_async(
                response=response_for_scoring,
                auxiliary_scorers=self._auxiliary_scorers,
                objective_scorers=[self._objective_scorer] if self._objective_scorer else [],
                role_filter="assistant",
                task=objective,
                skip_on_error=True,
            )

            # Extract objective score
            objective_scores = scoring_results["objective_scores"]
            if objective_scores:
                node.scores["objective"] = objective_scores[0]
                score_value = self._normalize_score_to_float(objective_scores[0])
                print(f"Objective Score: {score_value:.2f} (threshold: {self._successful_objective_threshold})")
                
                # Check if objective is achieved
                if score_value >= self._successful_objective_threshold:
                    print(f"🎯 OBJECTIVE ACHIEVED! Score {score_value:.2f} >= {self._successful_objective_threshold}")
                    objective_achieved = True

            # Extract auxiliary scores
            auxiliary_scores = scoring_results["auxiliary_scores"]
            for score in auxiliary_scores:
                scorer_name = score.scorer_class_identifier.get("__type__", "unknown")
                node.scores[scorer_name] = score
                print(f"Auxiliary Score ({scorer_name}): {score.get_value()}")

            return objective_achieved
                
        except Exception as e:
            self._logger.warning(f"Error scoring response: {e}")
            return False
    
    def _normalize_score_to_float(self, score: Optional[Score]) -> float:
        """
        Normalize any score to a float value between 0.0 and 1.0.
        Same implementation as TreeOfAttacks for consistency.

        Args:
            score: The score to normalize, or None.

        Returns:
            Float value between 0.0 and 1.0. Returns 0.0 if score is None.
        """
        if not score:
            return 0.0

        score_value = score.get_value()
        if isinstance(score_value, bool):
            return 1.0 if score_value else 0.0
        elif isinstance(score_value, (int, float)):
            return float(score_value)
        else:
            self._logger.warning(f"Unexpected score value type: {type(score_value)} with value: {score_value}")
            return 0.0
    
    def _create_attack_result(self, context: MultiBranchAttackContext) -> MultiBranchAttackResult:
        """
        Create the attack result.
        
        Args:
            context (MultiBranchAttackContext): The attack context.
            
        Returns:
            MultiBranchAttackResult: The attack result.
        """
        # Get the final conversation and response
        final_node = context.current_node
        final_conversation_id = final_node.conversation_id if final_node else ""
        
        # Get last response
        last_response = None
        last_score = None
        if final_node and final_node.last_target_response:
            last_response = final_node.last_target_response
            if "objective" in final_node.scores:
                last_score = final_node.scores["objective"]
        
        # Calculate tree metrics
        max_depth = max((node.depth for node in context.all_nodes), default=0)
        
        # Create conversation tree summary for analysis
        conversation_tree = None
        if context.root_node:
            conversation_tree = self._build_tree_summary(context.root_node)
        
        # Determine attack outcome based on final score
        outcome = AttackOutcome.UNDETERMINED
        outcome_reason = "Human-guided conversation completed"
        
        if last_score:
            normalized_score = self._normalize_score_to_float(last_score)
            if normalized_score >= self._successful_objective_threshold:
                outcome = AttackOutcome.SUCCESS
                outcome_reason = f"Objective achieved with score {normalized_score:.2f} >= {self._successful_objective_threshold}"
            else:
                outcome = AttackOutcome.FAILURE
                outcome_reason = f"Objective not achieved. Final score {normalized_score:.2f} < {self._successful_objective_threshold}"
        
        # add all conversations other than the final returned conversation as a related_conversation
        for node in context.all_nodes:
            if node.conversation_id != final_conversation_id:
                context.related_conversations.add(
                    ConversationReference(
                        conversation_id=node.conversation_id,
                        conversation_type=ConversationType.PRUNED
                    )
                )
        
        return MultiBranchAttackResult(
            attack_identifier=self.get_identifier(),
            conversation_id=final_conversation_id,
            objective=context.objective,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=context.total_turns,
            last_response=last_response,
            last_score=last_score,
            related_conversations=context.related_conversations,
            total_turns=context.total_turns,
            total_branches=context.total_branches,
            max_depth_reached=max_depth,
            final_node_id=final_node.node_id if final_node else "",
            conversation_tree=conversation_tree
        )
    
    def _build_tree_summary(self, node: ConversationNode) -> Dict:
        """
        Build a summary of the conversation tree for analysis.
        
        Args:
            node (ConversationNode): The root node to start from.
            
        Returns:
            Dict: A nested dictionary representing the tree structure.
        """
        summary = {
            "node_id": node.node_id,
            "depth": node.depth,
            "turn_count": node.turn_count,
            "last_prompt": node.last_user_prompt,
            "response": node.last_target_response,
            "scores": {name: score.get_value() for name, score in node.scores.items()},
            "children": [self._build_tree_summary(child) for child in node.children_nodes]
        }
        return summary
    
    @overload
    async def execute_async(
        self,
        *,
        objective: str,
        memory_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> MultiBranchAttackResult:
        """
        Execute the human multi-turn attack asynchronously.
        
        Args:
            objective (str): The objective of the attack.
            memory_labels (Optional[Dict[str, str]]): Memory labels for the attack context.
            **kwargs: Additional parameters for the attack.
            
        Returns:
            MultiBranchAttackResult: The result of the attack execution.
        """
        ...
    
    @overload
    async def execute_async(self, **kwargs) -> MultiBranchAttackResult: ...
    
    async def execute_async(self, **kwargs) -> MultiBranchAttackResult:
        """Execute the attack strategy asynchronously with the provided parameters."""
        return await super().execute_async(**kwargs)
