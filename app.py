def handle_kano_response(feat_obj):
    """Handle a single feature response."""
    # Ensure essential keys exist
    required_keys = ["feature", "when_present", "when_absent"]
    if not all(key in feat_obj for key in required_keys):
        return None  # Skip invalid entries

    # Get the values for present and absent ratings
    f = feat_obj["when_present"]
    d = feat_obj["when_absent"]

    # Check if the values are numeric or map them to the correct ratings
    rating_map = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, 
        "I like it": 1, "I expect it": 2, "I am indifferent": 3, 
        "I can live with it": 4, "I dislike it": 5
    }

    f = rating_map.get(str(f), f)
    d = rating_map.get(str(d), d)

    # Classify the Kano model
    classification = classify_kano(f, d)

    return {
        "Feature": feat_obj["feature"],
        "Present": f,
        "Absent": d,
        "Net Score": f - d,
        "Classification": classification
    }

# For each response, handle feature entries
for resp in kano_responses:
    try:
        parsed_json = json.loads(resp)
        
        # Show an example response for debugging
        if not debug_response_shown:
            st.write("📌 Debug: Example API Response", parsed_json)
            debug_response_shown = True

        if "features" not in parsed_json:
            st.warning(f"Missing 'features' in response: {parsed_json}")
            continue  # Skip if 'features' is missing

        for feat_obj in parsed_json.get("features", []):
            result = handle_kano_response(feat_obj)
            if result:
                classifications.append(result)
            else:
                st.warning(f"Skipping invalid entry: {feat_obj}")
                
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")

# Display results if valid classifications exist
if classifications:
    kano_df = pd.DataFrame(classifications)
    kano_df.index = kano_df.index + 1  # Fix indexing
    st.dataframe(kano_df)

    csv = kano_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Kano Results", data=csv, file_name="kano_results.csv", mime="text/csv")
else:
    st.warning("No valid Kano classifications found. Ensure survey responses are properly formatted.")
