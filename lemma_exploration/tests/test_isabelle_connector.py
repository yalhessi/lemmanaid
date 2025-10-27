from isabelle_connector.isabelle_connector import IsabelleConnector, parse_ml_value, temp_theory

import nest_asyncio
nest_asyncio.apply()


def test_echo():
    isabelle = IsabelleConnector(name="test", working_directory=".")
    responses = isabelle._client.echo("Hello World")
    response = responses[-1].response_body
    assert response == '"Hello World"'
    
    
def test_use_thy():
    isabelle = IsabelleConnector(name="test", working_directory=".")
    query = 'ML\<open> let val res = "Hello, World!" in res end \<close>'
    test_thy = temp_theory(
            working_directory=".",
            queries=[query],
            imports=[],
            name="Test",
        )
    
    result = isabelle.use_theory(
        test_thy,
        rm_after=True,
    )
    assert result == {"Test": ["Hello, World!"]}
