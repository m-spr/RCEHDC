-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY XoringPopCtrl IS
    GENERIC (
        n           : INTEGER := 10;   -- Bit width of popCounters (ceiling log2(featureSize))
        featureSize : INTEGER := 700   -- Number of loops for popCounts
    );
    PORT (
        clk       : IN  STD_LOGIC;
        rst       : IN  STD_LOGIC;
        run       : IN  STD_LOGIC;
        rundegi   : OUT STD_LOGIC;
        update    : OUT STD_LOGIC;
        doneI     : OUT STD_LOGIC;
        doneII    : OUT STD_LOGIC;
        ready_M   : OUT STD_LOGIC
    );
END ENTITY XoringPopCtrl;

ARCHITECTURE ctrl OF XoringPopCtrl IS

    -- PopCount Component Declaration
    COMPONENT popCount IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of popCounters (LOG2(featureSize))
        );
        PORT (
            clk  : IN  STD_LOGIC;
            rst  : IN  STD_LOGIC;
            en   : IN  STD_LOGIC;
            dout : OUT STD_LOGIC_VECTOR(lenPop-1 DOWNTO 0)
        );
    END COMPONENT;

    -- FSM States
    TYPE state IS (init, sum, fin, fin1);
    SIGNAL ns, ps : state;

    -- Internal Signals
    SIGNAL runPOP, rstPop : STD_LOGIC;
    SIGNAL count          : STD_LOGIC_VECTOR(n-1 DOWNTO 0);
    CONSTANT checker      : STD_LOGIC_VECTOR(n-1 DOWNTO 0) := STD_LOGIC_VECTOR(to_UNSIGNED(featureSize, n));


BEGIN

    -- Synchronous State Update Process
    PROCESS(clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF (rst = '1') THEN
                ps <= init; 
            ELSE  
                ps <= ns;  
            END IF;
        END IF;
    END PROCESS;

    -- Next State Logic Process
    PROCESS (ps, run, count)
    BEGIN
        -- Default values
        runPOP  <= '0';
        rstPop  <= '0';
        doneI   <= '0';
        doneII  <= '0';
        rundegi <= '0';
        ready_M <= '1';

        CASE ps IS 
            WHEN init =>
                rstPop <= '1';  
                IF run = '1' THEN
                    ns      <= sum;
                    rundegi <= '1';
                    runPOP  <= '1';
                    rstPop  <= '0';
                ELSE
                    ns <= init;
                END IF;

            WHEN sum =>
                IF count = checker THEN
                    ns <= fin1;
                ELSE
                    runPOP  <= '1';
                    rundegi <= '1';
                    ns      <= sum;
                END IF;

            WHEN fin1 =>
                ready_M <= '0';
                doneI   <= '1';
                rstPop  <= '1';
                ns      <= fin;

            WHEN fin =>
                rstPop  <= '1';
                ready_M <= '0';
                doneII  <= '1';
                ns      <= init;

            WHEN OTHERS =>
                ns <= init;
        END CASE;
    END PROCESS;

    -- Pop Counter Instance with Explicit Generic Mapping
    pop : popCount
        GENERIC MAP (
            lenPop => n
        )
        PORT MAP (
            clk  => clk,
            rst  => rstPop,
            en   => runPOP,
            dout => count
        );

    -- Output Assignment
    update <= runPOP;

END ARCHITECTURE ctrl;
