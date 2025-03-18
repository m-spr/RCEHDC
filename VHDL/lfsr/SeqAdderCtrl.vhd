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

ENTITY SeqAdderCtrl IS
    GENERIC (
        ceilingLogPop : INTEGER := 3;  -- Ceiling log2 of number of pop counters
        nPop         : INTEGER := 8    -- Number of pop counters
    );
    PORT (
        clk         : IN  STD_LOGIC;
        rst         : IN  STD_LOGIC;
        run         : IN  STD_LOGIC;
        reg1Update  : OUT STD_LOGIC;
        reg1rst     : OUT STD_LOGIC;
        reg2Update  : OUT STD_LOGIC;
        reg2rst     : OUT STD_LOGIC;
        muxSel      : OUT STD_LOGIC_VECTOR(ceilingLogPop DOWNTO 0)
    );
END ENTITY SeqAdderCtrl;

ARCHITECTURE ctrl OF SeqAdderCtrl IS

    -- Component Declaration
    COMPONENT popCount IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of output popCounters
        );
        PORT (
            clk   : IN  STD_LOGIC;
            rst   : IN  STD_LOGIC;
            en    : IN  STD_LOGIC;
            dout  : OUT STD_LOGIC_VECTOR(lenPop-1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signals
    SIGNAL count    : STD_LOGIC_VECTOR(ceilingLogPop DOWNTO 0);
    TYPE state IS (init, sum);
    SIGNAL ns, ps   : state;
    SIGNAL countEn, countRst : STD_LOGIC;

BEGIN

    -- Synchronous State Update Process
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF (rst = '1') THEN
                ps <= init;
            ELSE
                ps <= ns;
            END IF;
        END IF;
    END PROCESS;

    -- Next State Logic
    PROCESS (ps, run, count)
    BEGIN
        -- Default values
        countEn    <= '0';
        countRst   <= '0';
        reg1Update <= '0';
        reg1rst    <= '0';
        reg2Update <= '0';

        CASE ps IS
            WHEN init =>
                countRst <= '1';
                reg1rst  <= '1';

                IF run = '1' THEN
                    ns <= sum;
                ELSE
                    ns <= init;
                END IF;

            WHEN sum =>
                IF count = STD_LOGIC_VECTOR(to_UNSIGNED(nPop, count'length)) THEN
                    countRst <= '1';
                    IF run = '1' THEN
                        ns <= sum;
                        reg1rst <= '1';
                        reg2Update <= '1';
                    ELSE
                        ns <= init;
                        reg2Update <= '1';
                    END IF;
                ELSE
                    countEn    <= '1';
                    reg1Update <= '1';
                    ns         <= sum;
                END IF;

            WHEN OTHERS =>
                ns <= init;
        END CASE;
    END PROCESS;

    -- Direct reset assignment
    reg2rst <= rst;

    -- Pop Counter Instance with explicit generic mapping
    sel : popCount 
        GENERIC MAP (
            lenPop => ceilingLogPop + 1
        )
        PORT MAP (
            clk  => clk,
            rst  => countRst,
            en   => countEn,
            dout => count
        );

    -- Output assignment
    muxSel <= count;

END ARCHITECTURE ctrl;
