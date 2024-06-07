import os
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

from typing import Annotated

class MarkdownSearchEngine:
    def __init__(self, index_dir, docs_dir):
        self.index_dir = index_dir
        self.docs_dir = docs_dir
        self.schema = Schema(filename=ID(stored=True), content=TEXT)
        self.ix = None
        self.current_document = None
        self.initialize_index()

    def initialize_index(self):
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
        
        if exists_in(self.index_dir):
            self.ix = open_dir(self.index_dir)
        else:
            self.create_index()

    def create_index(self):
        self.ix = create_in(self.index_dir, self.schema)
        writer = self.ix.writer()

        for filename in os.listdir(self.docs_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    writer.add_document(filename=filename, content=content)
        
        writer.commit()

    def search_(self, query_str):
        with self.ix.searcher() as searcher:
            query = QueryParser("content", self.ix.schema).parse(query_str)
            results = searcher.search(query)
            if results:
                return [result['filename'] for result in results]
            else:
                return []
           
    def read_doc(self, filename):
        filepath = os.path.join(self.docs_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return lines[0:10]

    def search(self, query_str: Annotated[str, "The query to search for, no more than two words."]):
        """
        Search for a query in the index. Saves the first document that was found and returs the first lines of the document.
        The index is a Whoosh index over a collection of documents with various information.
        Returns None if no document was found.
        """
        filenames = self.search_(query_str)
        if filenames:
            self.current_document = filenames[0]
            lines = self.read_doc(self.current_document)
            return lines
        else:
            self.current_document = None
            return "No document found."

    def lookup(self, word: Annotated[str, "The word to lookup in the current document."]):
        """
        Lookup a word in the current document and return the lines that contain the word.
        """
        if not self.current_document:
            return "No document is currently selected."

        filepath = os.path.join(self.docs_dir, self.current_document)
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if word in line:
                start = max(0, i - 1)
                end = min(len(lines), i + 6)
                surrounding_lines = lines[start:end]
                return ''.join(surrounding_lines).strip()
        
        return "Word not found in the current document."

# Example usage
if __name__ == "__main__":
    # Directories
    index_dir = "indexdir"
    docs_dir = "docs"

    # Initialize the search engine
    search_engine = MarkdownSearchEngine(index_dir, docs_dir)

    # Search for keywords
    query = "oxygen"
    results = search_engine.search(query)

    # Display results
    if results:
        print("Documents containing the query:")
        for filename in results:
            print(filename)
    else:
        print("No documents found containing the query.")
    
    # Lookup a word in the current document
    if search_engine.current_document:
        word = 'atomic weight'
        lines = search_engine.lookup(word)
        print(f"First occurence of '{word}':\n{lines}")
    else:
        print("No document is currently selected for lookup.")
