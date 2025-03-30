# Check if kano_responses is not None and contains data
if kano_responses:
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
else:
    st.warning("No Kano responses found. Please ensure the survey was completed successfully.")

# Display results if valid classifications exist
if classifications:
    kano_df = pd.DataFrame(classifications)
    kano_df.index = kano_df.index + 1  # Fix indexing
    st.dataframe(kano_df)

    csv = kano_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Kano Results", data=csv, file_name="kano_results.csv", mime="text/csv")
else:
    st.warning("No valid Kano classifications found. Ensure survey responses are properly formatted.")
