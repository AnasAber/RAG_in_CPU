STRUCTURED_CV_RESUME_TEMPLATE = """
You are an expert resume analyzer. Your task is to structure the following resume content into clear, comprehensive sections. Focus on extracting and organizing information into these key areas:
- The name of the candidate

1. Professional Summary
2. Core Competencies
3. Technical Skills
4. Work Experience
5. Projects
6. Education
7. Certifications
8. Achievements

For each section:
- Provide a detailed summary, guarding the essential details, do not lose a lot of content.
- Use bullet points for listing items.
- Highlight key technologies, tools, and methodologies.
- Quantify achievements and impacts where possible.

Resume Content:
{context}

Now, structure the above resume content into the specified sections:

Professional Summary:

Core Competencies:

Technical Skills:

Work Experience:

Projects:

Education:

Certifications:

Achievements:

Additional Notes:
- Identify any unique or standout qualities in the resume.
- Note any potential gaps or areas for improvement.

Remember to maintain a professional tone and focus on clarity and relevance in your structuring.
"""