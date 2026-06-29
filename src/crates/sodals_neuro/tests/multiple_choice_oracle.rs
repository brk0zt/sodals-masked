#[cfg(test)]
mod multiple_choice_tests {
    use sodals_neuro::semantic_parser::{SyntaxOracle, ChoiceConstraint};
    use sodals_neuro::semantic_network::ResonanceNetwork;
    use candle_core::{Tensor, Device};

    #[test]
    fn test_choice_constraint_creation() {
        let choices = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
        let constraint = ChoiceConstraint::new(choices.clone());

        assert_eq!(constraint.num_choices(), 4);
        assert_eq!(constraint.choices, choices);
        assert!(!constraint.is_active);
    }

    #[test]
    fn test_choice_constraint_activation() {
        let mut constraint = ChoiceConstraint::new(vec!["yes".to_string(), "no".to_string()]);
        
        assert!(!constraint.is_active);
        constraint.activate();
        assert!(constraint.is_active);
        constraint.deactivate();
        assert!(!constraint.is_active);
    }

    #[test]
    fn test_validate_choice_exact_match() {
        let choices = vec!["Apple".to_string(), "Banana".to_string(), "Cherry".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Exact match
        assert_eq!(oracle.validate_choice("Apple", &constraint), Some(0));
        assert_eq!(oracle.validate_choice("Banana", &constraint), Some(1));
        assert_eq!(oracle.validate_choice("Cherry", &constraint), Some(2));
    }

    #[test]
    fn test_validate_choice_case_insensitive() {
        let choices = vec!["Yes".to_string(), "No".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Case-insensitive
        assert_eq!(oracle.validate_choice("yes", &constraint), Some(0));
        assert_eq!(oracle.validate_choice("YES", &constraint), Some(0));
        assert_eq!(oracle.validate_choice("no", &constraint), Some(1));
        assert_eq!(oracle.validate_choice("NO", &constraint), Some(1));
    }

    #[test]
    fn test_validate_choice_with_whitespace() {
        let choices = vec!["Option A".to_string(), "Option B".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // With extra whitespace
        assert_eq!(oracle.validate_choice("  Option A  ", &constraint), Some(0));
        assert_eq!(oracle.validate_choice("Option B", &constraint), Some(1));
    }

    #[test]
    fn test_validate_choice_prefix_match() {
        let choices = vec!["Answer: A".to_string(), "Answer: B".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Prefix match
        assert_eq!(oracle.validate_choice("Answer: A is correct", &constraint), Some(0));
        assert_eq!(oracle.validate_choice("Answer: B is the answer", &constraint), Some(1));
    }

    #[test]
    fn test_validate_choice_inactive_constraint() {
        let choices = vec!["Yes".to_string(), "No".to_string()];
        let constraint = ChoiceConstraint::new(choices); // Not activated

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // When inactive, always returns None
        assert_eq!(oracle.validate_choice("Yes", &constraint), None);
    }

    #[test]
    fn test_validate_choice_invalid_input() {
        let choices = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Invalid choice
        assert_eq!(oracle.validate_choice("D", &constraint), None);
        assert_eq!(oracle.validate_choice("Maybe", &constraint), None);
    }

    #[test]
    fn test_score_choices() -> candle_core::Result<()> {
        let choices = vec!["True".to_string(), "False".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Create a simple context vector with correct dtype
        let device = Device::Cpu;
        let h_context = Tensor::randn(0f32, 1.0, (128,), &device)?;

        let scores = oracle.score_choices(&h_context, &mut constraint)?;

        // Should have 2 scores (one for each choice)
        assert_eq!(scores.len(), 2);
        
        // Scores should be numeric and reasonable (non-NaN)
        for score in scores.iter() {
            assert!(score.is_finite(), "Score should be finite");
        }

        Ok(())
    }

    #[test]
    fn test_enforce_choice_constraint() -> candle_core::Result<()> {
        let choices = vec!["Option 1".to_string(), "Option 2".to_string(), "Option 3".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();
        constraint.choice_coherence = vec![0.1, 0.8, 0.3]; // Option 2 is best

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Enforce constraint with no generated text - should pick highest-scoring choice
        let selected = oracle.enforce_choice_constraint(&constraint, None)?;
        assert_eq!(selected, 1); // Option 2 (index 1) has highest coherence

        // Enforce constraint with valid generated text - should validate it
        let selected = oracle.enforce_choice_constraint(&constraint, Some("Option 1"))?;
        assert_eq!(selected, 0); // Option 1 is valid, so return its index

        Ok(())
    }

    #[test]
    fn test_suggest_valid_choice() {
        let choices = vec!["Choice A".to_string(), "Choice B".to_string(), "Choice C".to_string()];
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();
        constraint.choice_coherence = vec![0.5, 0.9, 0.3]; // Choice B is best

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        let suggestion = oracle.suggest_valid_choice(&constraint);
        assert_eq!(suggestion, Some("Choice B".to_string()));
    }

    #[test]
    fn test_suggest_valid_choice_inactive() {
        let choices = vec!["Yes".to_string(), "No".to_string()];
        let constraint = ChoiceConstraint::new(choices); // Not activated

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        let suggestion = oracle.suggest_valid_choice(&constraint);
        assert_eq!(suggestion, None);
    }

    #[test]
    fn test_multiple_choice_arc_scenario() {
        // Simulate an ARC multiple-choice task
        let choices = vec![
            "Transform: rotate 90°".to_string(),
            "Transform: flip horizontal".to_string(),
            "Transform: flip vertical".to_string(),
            "No transformation".to_string(),
        ];
        
        let mut constraint = ChoiceConstraint::new(choices);
        constraint.activate();

        let network = ResonanceNetwork::new(128);
        let oracle = SyntaxOracle::new(0.5, &network);

        // Test various model outputs
        let output1 = "Transform: rotate 90°";
        assert_eq!(oracle.validate_choice(output1, &constraint), Some(0));

        let output2 = "Transform: flip horizontal is the answer";
        assert_eq!(oracle.validate_choice(output2, &constraint), Some(1));

        let output3 = "Transform: flip vertical is correct";
        assert_eq!(oracle.validate_choice(output3, &constraint), Some(2));

        let output4 = "No transformation";
        assert_eq!(oracle.validate_choice(output4, &constraint), Some(3));
    }
}
