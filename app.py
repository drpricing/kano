st.write("### Kano Evaluations")
kano_responses = st.session_state.results["responses"]

# Kano classification rules
rating_map = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, 
    "I like it": 1, "I expect it": 2, "I am indifferent": 3, 
    "I can live with it": 4, "I dislike it": 5
}

def classify_kano(f, d):
    if f == 1 and d >= 4:
        return "Excitement"
    elif f == 2 and d == 5:
        return "Must-Have"
    elif f == 3 and d == 3:
        return "Indifferent"
    else:
        return "Expected"

classifications = []
debug_response_shown = False  # To only show one debug response

for resp in kano_responses:
    try:
        parsed_json = json.loads(resp)

        # Show an example response for debugging
        if not debug_response_shown:
            st.write("📌 Example Kano API Response:", parsed_json)
            debug_response_shown = True

        for feat_obj in parsed_json.get("features", []):
            if "feature" not in feat_obj or "when_present" not in feat_obj or "when_absent" not in feat_obj:
                st.warning(f"Skipping invalid entry: {feat_obj}")
                continue  # Skip this entry if essential keys are missing

            f = rating_map.get(str(feat_obj["when_present"]), feat_obj["when_present"])
            d = rating_map.get(str(feat_obj["when_absent"]), feat_obj["when_absent"])
            classification = classify_kano(f, d)

            classifications.append({
                "Feature": feat_obj["feature"],
                "Present": f,
                "Absent": d,
                "Net Score": f - d,
                "Classification": classification
            })
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
