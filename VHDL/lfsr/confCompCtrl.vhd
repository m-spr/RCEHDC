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

ENTITY confCompCtrl IS
    GENERIC (
        n   : INTEGER := 10;  -- Number of classes
        lgn : INTEGER := 4    -- Log2(n) for class indexing
    );
    PORT (
        clk, rst   : IN  STD_LOGIC;
        run        : IN  STD_LOGIC;
        runOut     : OUT STD_LOGIC;
        done       : OUT STD_LOGIC;
        TLAST_S    : OUT STD_LOGIC;
        TVALID_S   : OUT STD_LOGIC;
        pointer    : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0)  -- Supports up to 16 classes (4-bit index)
    );
END ENTITY confCompCtrl;

ARCHITECTURE ctrl OF confCompCtrl IS

    -- Component Declaration
    COMPONENT popCount IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of output popCounters
        );
        PORT (
            clk  : IN  STD_LOGIC;
            rst  : IN  STD_LOGIC;
            en   : IN  STD_LOGIC;
            dout : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signals
    SIGNAL count   : STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
    SIGNAL countEn, countRst : STD_LOGIC;

    -- FSM State Declaration
    TYPE state IS (init, counting, busshand);
    SIGNAL ns, ps : state;

BEGIN

    -- **Synchronous FSM State Update**
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                ps <= init;
            ELSE  
                ps <= ns;
            END IF;
        END IF;
    END PROCESS;

    -- **FSM Next State & Output Logic**
    PROCESS (ps, run, count)
    BEGIN
        -- Default Signal Values
        runOut   <= '0';
        countRst <= '0';
        countEn  <= '0';  
        done     <= '0';
        TLAST_S  <= '0';  
        TVALID_S <= '0';

        CASE ps IS 
            WHEN init =>
                countRst <= '1';  -- Reset the counter
                runOut   <= '1';  -- Enable operation
                IF run = '1' THEN
                    ns <= counting;
                ELSE
                    ns <= init;
                END IF;

            WHEN counting =>
                IF count = STD_LOGIC_VECTOR(to_UNSIGNED(n - 1, lgn)) THEN  -- Properly handle last count
                    done <= '1';
                    IF run = '1' THEN
                        countRst <= '1';
                        runOut   <= '1';
                        ns       <= counting;
                    ELSE
                        ns       <= busshand;
                    END IF;
                ELSE 
                    countEn <= '1';  -- Increment counter
                    ns      <= counting;
                END IF;

            WHEN busshand =>
                TLAST_S  <= '1';
                TVALID_S <= '1';
                ns       <= init;

            WHEN OTHERS =>
                ns <= init;
        END CASE;
    END PROCESS;

    -- **Pop Count Counter Instantiation**
    sel : popCount 
        GENERIC MAP (
            lenPop => lgn
        )
        PORT MAP (
            clk  => clk,
            rst  => countRst,
            en   => countEn,
            dout => count
        );

    -- Assign counter value to pointer
    pointer <= count;

END ARCHITECTURE ctrl;
