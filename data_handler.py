import os
from owlready2 import get_ontology, OwlReadyOntologyParsingError
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
from SPARQLWrapper import SPARQLWrapper

def load_ontology(file_path):
    """
    Nạp file ontology và trả về đối tượng ontology.
    Trả về None nếu có lỗi trong quá trình nạp.
    """
    try:
        # Kiểm tra xem đường dẫn file có tồn tại không
        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}")
            return None

        # Nạp ontology từ đường dẫn file
        onto = get_ontology(f"{file_path}")
        print(f"Đã nạp ontology thành công từ file: {file_path}")
        return onto
        
    except OwlReadyOntologyParsingError as e:
        print(f"Lỗi cú pháp Ontology: {e}")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định khi nạp ontology: {e}")
        return None

def load_authors_list(onto_data):
    """Trích xuất danh sách tác giả từ ontology."""
    onto_data.load()
    if onto_data is None:
        print("Không load được ontology")
        return []
        
    authors = []
    # Lấy danh sách tất cả các thực thể (individuals) của lớp "Author"
    for cls in onto_data.classes():
        if cls.name == "Author":
            Authorclass = cls
            break
    authors = Authorclass.instances()
    print(len(authors))
    # Kiểm tra xem danh sách tác giả có rỗng không
    if not authors:
        print("Không có tác giả nào được tìm thấy.")
        return []
    
    return authors
    #return ['Test']
