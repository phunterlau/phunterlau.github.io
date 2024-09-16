# Building a Lightweight Financial Agent: A Flexible Approach to Tool Use and Orchestration

In the rapidly evolving field of AI agents, there's a growing trend towards complex frameworks and libraries. However, for many practical applications, a simpler, more flexible approach can be just as effective. This blog post introduces a lightweight financial agent framework that demonstrates how powerful tool use and orchestration can be achieved without relying on heavy libraries like LangChain or LlamaIndex or CrewAI etc. 

For most cases, one only needs "tool using" and "orchestration", so why so complex? Check out the code [here](https://github.com/phunterlau/tiny-financial-agent).

![Red Panda financial analysis](/images/fin-analysis.jpeg)

## The Power of Simplicity and Flexibility

Our framework consists of just three main components: a driver, tools, and orchestration functions. This simplicity inspired by functional programming offers several advantages:

1. Easy to understand and modify
2. Not dependent on external packages beyond basic Python libraries and an API for language model interactions
3. Flexible enough to handle both simple and complex queries
4. Full control over the agent's behavior, making it easier to adapt to specific use cases

The core of our framework lies in its atomic tools and orchestration functions. Let's explore how these components work together to create a flexible and powerful financial analysis agent. In this approach, human define a few orchestration patterns and how each pattern calls for tools, and LLM can map each question to one or more patterns to solve the problem. Here it is a sector analysis example where user asks a complex question "Considering the current economic climate, analyze the banking sector trends for the next 2 years and provide a comparative strategic investment analysis for JPMorgan Chase (JPM) and Bank of America (BAC)." and the agent understands it, maps it to a sector analysis orchestration flow, pick up the right tools, and summarize the results:

![Sector analysis example](/images/sector-analysis.png)

## Atomic Tools: The Building Blocks

Atomic tools are the fundamental operations our agent can perform. In our financial agent, these include functions like `get_stock_price`, `get_company_financials`, and `get_income_statement`. Here's an example of how an atomic tool might be implemented:

```python
def get_stock_price(symbol: str) -> FinancialData:
    url = f"https://financialmodelingprep.com/api/v3/quote-order/{symbol}?apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()[0]
    return FinancialData(**data)
```

This function makes an API call to retrieve stock price data and returns it in a structured format. The simplicity of these atomic tools makes them easy to test, maintain, and extend.

## Orchestration: Connecting the Dots

While atomic tools are powerful, they often need to be combined in complex ways to perform meaningful analyses. This is where orchestration functions come in. Orchestration allows us to dynamically connect tools using chain-of-thought (CoT) reasoning, enabling more sophisticated analyses.

Let's look at two orchestration functions to illustrate the range of complexity possible within this framework:

1. A simple orchestration function: SectorAnalysis

```python
class SectorAnalysis(OrchestrationFunction):
    def gather_data(self, sector: str, top_n: int = 5) -> Dict[str, Any]:
        companies = get_top_companies(sector, top_n)
        sector_data = []
        for company in companies:
            financials = self.use_atomic_function('get_company_financials', company['symbol'])
            income = self.use_atomic_function('get_income_statement', company['symbol'])
            stock_price = self.use_atomic_function('get_stock_price', company['symbol'])
            sector_data.append({
                "symbol": company['symbol'],
                "name": financials.companyName,
                "market_cap": financials.marketCap,
                "revenue": income.revenue,
                "net_income": income.net_income,
                "pe_ratio": stock_price.PE
            })
        return {"sector": sector, "top_n": top_n, "companies": sector_data}
```

This function performs a straightforward analysis of top companies in a given sector. It uses atomic functions in a predetermined sequence to gather and structure data.

2. A complex orchestration function: CompanyComparativeAnalysis

```python
class CompanyComparativeAnalysis(OrchestrationFunction):
    def gather_data(self, symbol1: str, symbol2: str, time_horizon: str) -> Dict[str, Any]:
        company1_data = self._gather_company_data(symbol1)
        company2_data = self._gather_company_data(symbol2)
        
        return {
            "company1": company1_data,
            "company2": company2_data,
            "time_horizon": time_horizon
        }
    
    def _gather_company_data(self, symbol: str) -> Dict[str, Any]:
        financials = self.use_atomic_function('get_company_financials', symbol)
        income = self.use_atomic_function('get_income_statement', symbol)
        stock_price = self.use_atomic_function('get_stock_price', symbol)
        historical_data = self.use_atomic_function('get_historical_price_data', symbol)
        
        return {
            "symbol": symbol,
            "financials": financials,
            "income": income,
            "stock_price": stock_price,
            "historical_data": historical_data
        }

    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        return f"""
        Perform a comparative analysis of {data['company1']['symbol']} and {data['company2']['symbol']} over a {data['time_horizon']} time horizon.
        Include a competitive analysis and assessment of investment potential for both companies.
        
        Company 1 ({data['company1']['symbol']}) Data:
        {json.dumps(data['company1'], indent=2)}
        
        Company 2 ({data['company2']['symbol']}) Data:
        {json.dumps(data['company2'], indent=2)}
        
        Provide a comprehensive analysis covering:
        1. Competitive position of both companies
        2. Financial performance comparison
        3. Growth prospects over the {data['time_horizon']} time horizon
        4. Potential risks and opportunities
        5. Overall investment potential comparison
        """
```

This more complex function demonstrates how orchestration can adapt to different scenarios and gather a wider range of data. It shows how orchestration functions can implement more sophisticated logic to determine which tools to use and how to combine their outputs.

## The Power of Orchestration in Action

To truly appreciate the flexibility and power of our orchestration approach, let's examine how a complex query triggers the appropriate orchestration function:

Query: "Compare the investment potential of Microsoft (MSFT) and Google (GOOGL) over the next 3 years, including a competitive analysis of both companies."

This query would activate the `CompanyComparativeAnalysis` orchestration function:

```python
# CompanyComparativeAnalysis execution
company1_data = self._gather_company_data('MSFT')
company2_data = self._gather_company_data('GOOGL')

# For each company, the following atomic functions are called:
# - get_company_financials
# - get_income_statement
# - get_stock_price
# - get_historical_price_data

# The gathered data is then used to prepare a comprehensive prompt for the language model
```

This example showcases how our framework can handle complex queries by combining multiple atomic tools within a single, sophisticated orchestration function. It performs a comparative analysis, including competitive positioning and investment potential assessment for both companies over the specified time horizon.

## Flexibility in Action: The FunctionCallingAgent

The heart of our framework's flexibility lies in the `FunctionCallingAgent` class. This class determines which orchestration function to call based on the user's query. Here's a simplified version of its `chat` method:

```python
def chat(self, query: str) -> str:
    self.memory.append({"role": "user", "content": query})
    
    response = self.llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=self.memory,
        functions=[tool.model_dump() for tool in self.tools],
        function_call="auto"
    )
    
    if response.choices[0].message.function_call:
        function_call = response.choices[0].message.function_call
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        
        result = self.orchestration_functions[function_name].execute(**function_args)
        self.memory.append({"role": "function", "name": function_name, "content": str(result)})
    
    final_response = self.llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=self.memory
    )
    
    self.memory.append({"role": "assistant", "content": final_response.choices[0].message.content})
    return final_response.choices[0].message.content
```

This design allows the agent to dynamically select the most appropriate orchestration function based on the query's complexity and requirements.

## Conclusion: A Flexible Design Pattern

The agentic framework presented here is not just a collection of tools, but a design pattern for approaching complex problems. By separating atomic tools from orchestration functions and employing a flexible function-calling agent, we create a system that can easily adapt to new scenarios or be extended with new capabilities.

This approach also positions us well for future developments in AI. As more advanced chain-of-thought models become available, we can easily adapt our framework. We could use smaller, more efficient models for atomic tool use, reserving the more powerful CoT models for complex orchestration tasks.

In conclusion, while there's certainly a place for comprehensive agent frameworks, there's also value in understanding how to build lightweight, customizable agents from the ground up. This approach gives developers more control, better understanding of their agents' behavior, and the flexibility to adapt to new developments in AI technology.

The complete code for this financial agent example, along with additional documentation, can be found at [GitHub link](https://github.com/phunterlau/tiny-financial-agent). We encourage you to explore, adapt, and build upon this framework for your own projects.