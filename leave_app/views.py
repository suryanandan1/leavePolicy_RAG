from functools import lru_cache

from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from loaders import load_pdfs
from splitter import split_documents
from embeddings_store import create_vectorstore
from qa_chain import build_qa_chain
from employee_data import get_employee_by_id, signup_employee_excel
from auth_db import create_user, verify_user

import markdown


@lru_cache(maxsize=1)
def load_chain():
    """Load the existing RAG pipeline once and reuse it for all requests."""
    docs = load_pdfs(["data/leave.pdf"])
    split_docs = split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    return build_qa_chain(vectorstore)


def _get_employee_from_session(request):
    employee_id = request.session.get("employee_id")
    if not employee_id:
        return None
    return get_employee_by_id(employee_id)


def _build_full_query(employee_data, query):
    return f"""
Employee Information:
Employee ID: {employee_data['employee_id']}
Name: {employee_data['name']}
Grade/Band: {employee_data['grade']}
PL Taken: {employee_data['PL_taken']}
CL Taken: {employee_data['CL_taken']}
SL Taken: {employee_data['SL_taken']}

User Question:
{query}
"""


def _format_answer(answer):
    """
    Convert markdown answer into HTML.
    This makes markdown tables show as real HTML tables.
    """
    answer = answer.replace("**", "")

    return markdown.markdown(
        answer,
        extensions=["tables", "fenced_code"]
    )


@require_http_methods(["GET", "POST"])
def login_page(request):
    if request.method == "POST":
        employee_id = request.POST.get("employee_id", "").strip()
        password = request.POST.get("password", "").strip()

        if not employee_id or not password:
            messages.error(request, "Employee ID and password are required.")
            return redirect("login")

        if not verify_user(employee_id, password):
            messages.error(request, "Invalid Employee ID or Password.")
            return redirect("login")

        employee_data = get_employee_by_id(employee_id)
        if employee_data is None:
            messages.error(request, "Employee data not found in Excel.")
            return redirect("login")

        request.session["employee_id"] = employee_id
        request.session["chat_messages"] = []
        return redirect("chat")

    return render(request, "leave_app/login.html")


@require_http_methods(["GET", "POST"])
def signup_page(request):
    fetched_employee = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "fetch":
            fetch_id = request.POST.get("fetch_id", "").strip()
            fetched_employee = get_employee_by_id(fetch_id)

            if fetched_employee is None:
                messages.warning(
                    request,
                    "Employee not found in Excel. You can create a new signup."
                )
            else:
                messages.success(request, "Employee data fetched from Excel.")

        if action == "signup":
            employee_id = request.POST.get("employee_id", "").strip()
            password = request.POST.get("password", "").strip()
            name = request.POST.get("name", "").strip()
            grade = request.POST.get("grade", "").strip()
            joining_date = request.POST.get("joining_date", "").strip()
            pl_taken = int(request.POST.get("pl_taken") or 0)
            cl_taken = int(request.POST.get("cl_taken") or 0)
            sl_taken = int(request.POST.get("sl_taken") or 0)

            if not all([employee_id, password, name, grade]):
                messages.error(
                    request,
                    "Employee ID, password, name, and grade are required."
                )
                return redirect("signup")

            auth_success, auth_message = create_user(employee_id, password)
            if not auth_success:
                messages.error(request, auth_message)
                return redirect("signup")

            excel_success, excel_message = signup_employee_excel(
                employee_id,
                name,
                grade,
                joining_date,
                pl_taken,
                cl_taken,
                sl_taken
            )

            if not excel_success:
                messages.error(request, excel_message)
                return redirect("signup")

            messages.success(request, "Signup successful. Please login.")
            return redirect("login")

    return render(
        request,
        "leave_app/signup.html",
        {"fetched_employee": fetched_employee}
    )


@require_http_methods(["GET", "POST"])
def chat_page(request):
    employee_data = _get_employee_from_session(request)

    if employee_data is None:
        return redirect("login")

    chat_messages = request.session.get("chat_messages", [])

    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        if query:
            chat_messages.append({
                "role": "user",
                "content": query
            })

            full_query = _build_full_query(employee_data, query)

            try:
                response = load_chain().invoke({
                    "query": full_query
                })

                answer = response["result"]
                formatted_answer = _format_answer(answer)

            except Exception as exc:
                formatted_answer = f"Error while generating answer: {exc}"

            chat_messages.append({
                "role": "assistant",
                "content": formatted_answer
            })

            request.session["chat_messages"] = chat_messages
            request.session.modified = True

        return redirect("chat")

    return render(
        request,
        "leave_app/chat.html",
        {
            "employee": employee_data,
            "chat_messages": chat_messages
        }
    )


def logout_page(request):
    request.session.flush()
    return redirect("login")